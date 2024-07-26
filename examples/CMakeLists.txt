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

add_executable( undistort_images
    undistort_images.cpp
)
target_link_libraries( undistort_images ${OpenCV_LIBS} gflags )

add_executable( run_spherical_sfm
    spherical_sfm_tools.h spherical_sfm_tools.cpp
    spherical_sfm_io.h spherical_sfm_io.cpp
    run_spherical_sfm.cpp
)
target_link_libraries( run_spherical_sfm sphericalsfm )
target_link_libraries( run_spherical_sfm ${OpenCV_LIBS} gflags )
target_link_libraries( run_spherical_sfm GSL::gsl )
target_link_libraries( run_spherical_sfm ${GOPT_LIBRARIES} )
if ( OpenMP_FOUND )
    target_link_libraries( run_spherical_sfm OpenMP::OpenMP_CXX )
endif()
#target_compile_features( run_spherical_sfm PRIVATE cxx_alias_templates )

add_executable( run_spherical_sfm_uncalib
    colmap.h colmap.cpp
    spherical_sfm_tools.h spherical_sfm_tools.cpp
    spherical_sfm_io.h spherical_sfm_io.cpp
    run_spherical_sfm_uncalib.cpp
)
target_link_libraries( run_spherical_sfm_uncalib sphericalsfm )
target_link_libraries( run_spherical_sfm_uncalib ${OpenCV_LIBS} gflags )
target_link_libraries( run_spherical_sfm_uncalib GSL::gsl )
target_link_libraries( run_spherical_sfm_uncalib ${GOPT_LIBRARIES} )
target_link_libraries( run_spherical_sfm_uncalib ${SQLite3_LIBRARIES} )
if ( OpenMP_FOUND )
    target_link_libraries( run_spherical_sfm_uncalib OpenMP::OpenMP_CXX )
endif()

#include_directories( ${Boost_INCLUDE_DIRS} )
#add_executable( run_spherical_sfm_uncalib_mst
    #spherical_sfm_tools.h spherical_sfm_tools.cpp
    #spherical_sfm_tools_mst.h spherical_sfm_tools_mst.cpp
    #spherical_sfm_io.h spherical_sfm_io.cpp
    #run_spherical_sfm_uncalib_mst.cpp
#)
#target_link_libraries( run_spherical_sfm_uncalib_mst sphericalsfm )
#target_link_libraries( run_spherical_sfm_uncalib_mst ${OpenCV_LIBS} gflags )
#target_link_libraries( run_spherical_sfm_uncalib_mst ${Boost_LIBRARIES} )
#target_link_libraries( run_spherical_sfm_uncalib_mst OpenMP::OpenMP_CXX )

set( BUILD_STITCHER FALSE CACHE BOOL "Build stereo panorama stitcher app (requires Cuda)" )
if ( BUILD_STITCHER )
    add_executable( make_stereo_panorama
        stereo_panorama_tools.h stereo_panorama_tools.cpp
        make_stereo_panorama.cpp )
    target_link_libraries( make_stereo_panorama sphericalsfm gflags )
    add_executable( make_circle_views
        stereo_panorama_tools.h stereo_panorama_tools.cpp
        make_circle_views.cpp )
    target_link_libraries( make_circle_views sphericalsfm gflags )
endif ()

