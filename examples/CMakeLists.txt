
find_package( OpenMVG REQUIRED )

add_library( utils utils.h utils.cpp )
target_link_libraries( utils ${OpenCV_LIBS} )

add_executable( run_spherical_sfm
    feature_tracking.h feature_tracking.cpp
    #loop_closure.h loop_closure.cpp
    #rotation_averaging.h rotation_averaging.cpp
    #bundle_adjustment.h bundle_adjustment.cpp
    run_spherical_sfm.cpp
)
target_link_libraries( run_spherical_sfm sphericalsfm utils )
target_link_libraries( run_spherical_sfm OpenMVG::openMVG_multiview )
target_link_libraries( run_spherical_sfm ${OpenCV_LIBS} )
target_compile_features( run_spherical_sfm PRIVATE cxx_alias_templates )
target_compile_options( run_spherical_sfm PRIVATE ${OPENMVG_CFLAGS} )

set( BUILD_STITCHER FALSE CACHE BOOL "Build stereo panorama stitcher app (requires Cuda)" )
if ( BUILD_STITCHER )
    add_executable( make_stereo_panorama make_stereo_panorama.cpp )
    target_link_libraries( make_stereo_panorama sphericalsfm )
endif ()

