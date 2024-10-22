import pandas as pd
import os
import torch
import pytorch3d.renderer
from metric import camera_to_rel_deg, calculate_auc_np
import numpy as np
import glob
from scipy.spatial.transform import Rotation
from colmap_utils import read_cameras_from_sparse
import numpy as np

import argparse

def get_poses(path):
    cameras = read_cameras_from_sparse(path)
    cameras_dict = {}
    for camera in cameras:
        cameras_dict[camera['img_name']] = camera
    return cameras_dict

def colmap_to_pytorch3d(w2c):
    c2w = torch.inverse(w2c) # to c2w
    R, T = c2w[:3, :3], c2w[:3, 3:]
    R = torch.stack([-R[:, 0], -R[:, 1], R[:, 2]], 1) # from RDF to LUF for Rotation

    new_c2w = torch.cat([R, T], 1)
    w2c = torch.linalg.inv(torch.cat((new_c2w, torch.Tensor([[0,0,0,1]])), 0))
    R, T = w2c[:3, :3].permute(1, 0), w2c[:3, 3] # convert R to row-major matrix
    return R,T

def make_pytorch3d_cameras(poses,names):
    focal_length = torch.tensor([poses[name]['intrinsics'][0] for name in names]).float()
    cx = [poses[name]['intrinsics'][1] for name in names]
    cy = [poses[name]['intrinsics'][2] for name in names]
    principal_point = torch.tensor(np.stack([cx,cy],axis=-1)).float()
    R = []
    T = []
    for name in names:
        myR,myT = colmap_to_pytorch3d(poses[name]['w2c'])
        R.append(myR)
        T.append(myT)
    R = torch.stack(R,dim=0)
    T = torch.stack(T,dim=0)
    cameras = pytorch3d.renderer.cameras.PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        in_ndc=False,
        R=R,
        T=T,
        device='cpu')
    return cameras

def load_largest(path):
    poses = None
    try:
        poses = get_poses(path)
    except:
        pass
    if poses is None:
        paths = glob.glob(path+'/*')
        if len(paths) == 0:
           raise ValueError(f'could not find poses in {path}')
        lens = [len(get_poses(path)) for path in paths]
        poses = get_poses(paths[np.argmax(lens)])
    if poses is None:
       raise ValueError(f'could not find poses in {path}')
    return poses

def process_dataset(sfm_path,gt_path):
    sfm_poses = load_largest(sfm_path)
    gt_poses = load_largest(gt_path)

    names = [name for name in sfm_poses.keys() if name in gt_poses.keys()]

    sfm_focal = sfm_poses[names[0]]['intrinsics'][0]
    gt_focal = gt_poses[names[0]]['intrinsics'][0]
    focal_err = np.abs(sfm_focal-gt_focal)/gt_focal

    sfm_cameras = make_pytorch3d_cameras(sfm_poses,names)
    gt_cameras = make_pytorch3d_cameras(gt_poses,names)

    rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(sfm_cameras, gt_cameras, device='cpu', batch_size=1)
    rel_rangle_deg = rel_rangle_deg.detach().numpy()
    rel_tangle_deg = rel_tangle_deg.detach().numpy()

    return rel_rangle_deg, rel_tangle_deg, focal_err

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',required=True,help='path to directory containing scenes')
    parser.add_argument('--names',required=True,help='comma-separated list of dataset names')
    parser.add_argument('--pred',required=True,help='path within scene directory to predicted poses')
    parser.add_argument('--gt',required=True,help='path within scene directory to gt poses')
    args = parser.parse_args()

    def process_name(name):
        print(name)
        sfm_path = os.path.join(args.path,name,args.pred)
        gt_path = os.path.join(args.path,name,args.gt)
        return process_dataset(sfm_path,gt_path)
    
    res = list(map(process_name,args.names.split(',')))
    rel_rangle_deg,rel_tangle_deg,focal_err = zip(*res)
    rError = np.concatenate(rel_rangle_deg,axis=0)
    tError = np.concatenate(rel_tangle_deg,axis=0)
    fError = np.array(focal_err)

    res_dict = {}
    res_dict['Racc_5'] = np.mean(rError < 5) * 100
    res_dict['Racc_15'] = np.mean(rError < 15) * 100
    res_dict['Racc_30'] = np.mean(rError < 30) * 100
    
    res_dict['Tacc_5'] = np.mean(tError < 5) * 100
    res_dict['Tacc_15'] = np.mean(tError < 15) * 100
    res_dict['Tacc_30'] = np.mean(tError < 30) * 100
    
    res_dict['Auc_30'] = calculate_auc_np(rError, tError, max_threshold=30) * 100

    res_dict['focal'] = np.mean(fError) * 100
    
    df = pd.Series(res_dict)
    print(df.round(2))
    

