find_package(OpenMP)
find_package(gflags REQUIRED)
find_package(Boost REQUIRED)
find_package(GSL REQUIRED)

find_package( SQLite3 REQUIRED )
include_directories( ${SQLite3_INCLUDE_DIRS} )

find_package(Gopt REQUIRED)
if(GOPT_FOUND)
  message(STATUS "Found GOPT_INCLUDE_DIRs ${GOPT_INCLUDE_DIRS}")
  message(STATUS "Found GOPT_LINK_DIRs ${GOPT_LINK_DIRS}")
  message(STATUS "Found GOPT_LIBRARIES ${GOPT_LIBRARIES}")
else(GOPT_FOUND)
  message(FATAL "GOPT not found!")
endif(GOPT_FOUND)

include_directories(${GOPT_INCLUDE_DIRS})
link_directories(${GOPT_LINK_DIRS})

add_library( sphericalsfmtools
    colmap.h colmap.cpp
    spherical_sfm_tools.h spherical_sfm_tools.cpp
    spherical_sfm_io.h spherical_sfm_io.cpp
)
target_link_libraries( sphericalsfmtools sphericalsfm )
target_link_libraries( sphericalsfmtools ${OpenCV_LIBS} )
target_link_libraries( sphericalsfmtools GSL::gsl )
target_link_libraries( sphericalsfmtools ${GOPT_LIBRARIES} )
target_link_libraries( sphericalsfmtools ${SQLite3_LIBRARIES} )
if ( OpenMP_FOUND )
    target_link_libraries( sphericalsfmtools OpenMP::OpenMP_CXX )
endif()


add_executable( undistort_images undistort_images.cpp )
target_link_libraries( undistort_images ${OpenCV_LIBS} gflags )

add_executable( run_spherical_sfm run_spherical_sfm.cpp )
target_link_libraries( run_spherical_sfm sphericalsfmtools gflags )

add_executable( anms_feature_extractor anms_feature_extractor.cpp )
target_link_libraries( anms_feature_extractor sphericalsfmtools gflags )

add_executable( run_spherical_sfm_uncalib run_spherical_sfm_uncalib.cpp )
target_link_libraries( run_spherical_sfm_uncalib sphericalsfmtools gflags )


set( BUILD_STITCHER FALSE CACHE BOOL "Build stereo panorama stitcher app (requires Cuda)" )
if ( BUILD_STITCHER )
    add_executable( make_stereo_panorama make_stereo_panorama.cpp )
    target_link_libraries( make_stereo_panorama sphericalsfmtools gflags )
    add_executable( make_circle_views make_circle_views.cpp )
    target_link_libraries( make_circle_views sphericalsfmtools gflags )
endif ()

find_package( PoseLib REQUIRED ) 
find_package( opengv REQUIRED ) 
add_executable( test_spherical_relpose test_spherical_relpose.cpp )
target_link_libraries( test_spherical_relpose sphericalsfm PoseLib opengv )
