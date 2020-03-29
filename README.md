## Spherical Structure-from-Motion

![Elfin Forest](teaser.jpg "Elfin Forest")

Code for our papers:

Baker, L., S. Mills, S. Zollmann, and J. Ventura, "CasualStereo: Casual Capture of Stereo Panoramas with Spherical Structure-from-Motion", IEEE Conference on Virtual Reality and 3D User Interfaces (VR), 2020.

Ventura, J., "Structure from Motion on a Sphere", European Conference on Computer Vision (ECCV), 2016.

### Dependencies

* OpenCV 3+
* Ceres solver
* Eigen 3+
* [Polynomial](https://github.com/jonathanventura/polynomial)

### Usage

To run the spherical structure-from-motion pipeline:

    run_spherical_sfm -intrinsics <path to intrinsics> -video <path to video> -output <path to output>

The video path can be an image filename specifier such as %06d.png.

To make the stereo panoramas:

    make_stereo_panorama -intrinsics <path to intrinsics> -video <path to video> -output <path to output>

### Examples

You can view these example panoramas in a WebVR-compatible browser or headset.

* [Elfin Forest](webviewer/index.html?name=elfinforest)
* [Children's Garden](webviewer/index.html?name=childrensgarden)
* [Pismo Beach](webviewer/index.html?name=pismo)
* [Owheo Courtyard](webviewer/index.html?name=owheo)
* [Street](webviewer/index.html?name=street)

