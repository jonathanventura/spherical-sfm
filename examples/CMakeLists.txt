
add_executable( run_spherical_sfm
    spherical_sfm_tools.h spherical_sfm_tools.cpp
    spherical_sfm_io.h spherical_sfm_io.cpp
    run_spherical_sfm.cpp
)
target_link_libraries( run_spherical_sfm sphericalsfm )
target_link_libraries( run_spherical_sfm ${OpenCV_LIBS} )
target_compile_features( run_spherical_sfm PRIVATE cxx_alias_templates )
#target_link_libraries( run_spherical_sfm OpenMVG::openMVG_multiview )
#target_compile_options( run_spherical_sfm PRIVATE ${OPENMVG_CFLAGS} )

set( BUILD_STITCHER FALSE CACHE BOOL "Build stereo panorama stitcher app (requires Cuda)" )
if ( BUILD_STITCHER )
    add_executable( make_stereo_panorama
        stereo_panorama_tools.h stereo_panorama_tools.cpp
        make_stereo_panorama.cpp )
    target_link_libraries( make_stereo_panorama sphericalsfm )
endif ()

