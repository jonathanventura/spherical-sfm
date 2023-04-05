""" Converts a Spherical SFM reconstruction to the JSON format
    used by Instant Neural Graphic Primitives.

    Spherical SFM data should be organized as follows:
    base/
        calib.txt  # contains focal,cx,cy
        poses.txt  # contains world-to-camera transformations, images are indexed starting at 0
        images/    # contains images with filename pattern %06d.png, starting with 1
            000001.png 
            000002.png 
            000003.png 
            ...

    Some code adapted from:
    https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py
"""
import numpy as np
import cv2
from imageio import imread
from scipy.spatial.transform import Rotation
import os
import tqdm

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c)**2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa+ta*da+ob+tb*db) * 0.5, denom

def create_transforms(path,aabb_scale=16,keep_colmap_coords=False,stride=1):
    calib_path = os.path.join(path,'calib.txt')
    poses_path = os.path.join(path,'poses.txt')
    images_path = os.path.join(path,f'images')

    focal, cx, cy = np.loadtxt(calib_path)
    
    data = np.loadtxt(poses_path)

    data = data[::stride]


    up = np.zeros(3)
    frames = []
    pbar = tqdm.tqdm(total=len(data))
    for i,row in enumerate(data):
        index = int(row[0])
        image_path = '%s/%06d.jpg'%(images_path,index+1)

        if i == 0:
            image = imread(image_path)
            h,w = image.shape[:2]
        
        #jpg_path = '%s/%06d.jpg'%(images_path,index+1)
        #im = cv2.imread(image_path)
        #cv2.imwrite(jpg_path,im)
 
        t = np.reshape(row[1:4],(3,1))
        r = row[4:7]
        R = Rotation.from_rotvec(r).as_matrix()
        w2c = np.concatenate([R,t],axis=-1)
        w2c = np.concatenate([w2c,[[0,0,0,1]]],axis=0)

        c2w = np.linalg.inv(w2c)

        if not args.keep_colmap_coords:
            c2w[0:3,2] *= -1 # flip the y and z axis
            c2w[0:3,1] *= -1
            c2w = c2w[[1,0,2,3],:] # swap y and z
            c2w[2,:] *= -1 # flip whole world upside down

            up += c2w[0:3,1]


        frames.append({
            'file_path':'images/%06d.jpg'%(index+1),
            'sharpness':sharpness(image_path),
            'transform_matrix':c2w,
        })
        pbar.update(1)
    pbar.close()

    fl_x = focal
    fl_y = focal
    # tan(theta_x/2) = (W/2)/focal
    camera_angle_x = 2 * np.arctan2(w/2,fl_x)
    camera_angle_y = 2 * np.arctan2(h/2,fl_y)
    k1 = 0
    k2 = 0
    p1 = 0
    p2 = 0

    out = {
        'camera_angle_x':camera_angle_x,
        'camera_angle_y':camera_angle_y,
        'k1':k1,
        'k2':k2,
        'p1':p1,
        'p2':p2,
        'cx':cx,
        'cy':cy,
        'w':w,
        'h':h,
        'aabb_scale':aabb_scale,
        'frames':frames
    }

    nframes = len(out["frames"])

    if keep_colmap_coords:
        pass
        """
        flip_mat = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat) # flip cameras (it just works)
        """
    else:
        # don't keep colmap coords - reorient the scene to be easier to work with

        up = up / np.linalg.norm(up)
        print("up vector was", up)
        R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
        R = np.pad(R,[0,1])
        R[-1, -1] = 1

        for f in out["frames"]:
            f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

        # find a central point they are all looking at
        print("computing center of attention...")
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for f in out["frames"]:
            mf = f["transform_matrix"][0:3,:]
            for g in out["frames"]:
                mg = g["transform_matrix"][0:3,:]
                p, w = closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                if w > 0.01:
                    totp += p*w
                    totw += w
        totp /= totw
        print(totp) # the cameras are looking at totp
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] -= totp

        avglen = 0.
        for f in out["frames"]:
            avglen += np.linalg.norm(f["transform_matrix"][0:3,3])
        avglen /= nframes
        print("avg camera distance from origin", avglen)
        for f in out["frames"]:
            f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()
    print(nframes,"frames")

    return out

def create_downsampled_images(path):
    cmd = 'rm -rf ' + os.path.join(path,'images_4')
    os.system(cmd)
    cmd = 'mkdir -p ' + os.path.join(path,'images_4')
    os.system(cmd)
    cmd = 'cp -r ' + os.path.join(path,'images/*') + ' ' + os.path.join(path,'images_4/.')
    os.system(cmd)
    cmd = 'mogrify -resize 25% ' + os.path.join(path,'images_4/*.jpg')
    os.system(cmd)

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('path',help='path to spherical sfm data folder')
    parser.add_argument('--aabb_scale',type=int,default=16)
    parser.add_argument('--keep_colmap_coords',action='store_true')
    parser.add_argument('--stride',type=int,default=1,help='interval at which to sample images (2 means take every second image)')
    args = parser.parse_args()
    
    transforms = create_transforms(
        args.path,
        aabb_scale=args.aabb_scale,
        keep_colmap_coords=args.keep_colmap_coords,
        stride=args.stride)   

    json_path = os.path.join(args.path,'transforms.json')
    with open(json_path, "w") as f:
        json.dump(transforms, f, indent=2)
    
    create_downsampled_images(args.path)
 

