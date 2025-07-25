
cmake_minimum_required( VERSION 2.6 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()


project( spherical-sfm )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17 -Wall")

find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )
include_directories( ${OPENCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

include_directories( include )
add_library( sphericalsfm
    include/sphericalsfm/ray.h
    include/sphericalsfm/so3.h src/so3.cpp
    include/sphericalsfm/sparse.hpp
    include/sphericalsfm/sfm.h src/sfm.cpp
    include/sphericalsfm/sfm_types.h src/sfm_types.cpp
    include/sphericalsfm/estimator.h
    include/sphericalsfm/msac.h
    include/sphericalsfm/preemptive_ransac.h
    include/sphericalsfm/spherical_solvers.h src/spherical_solvers.cpp
    include/sphericalsfm/spherical_estimator.h src/spherical_estimator.cpp
    include/sphericalsfm/triangulation_estimator.h src/triangulation_estimator.cpp
    include/sphericalsfm/plane_estimator.h src/plane_estimator.cpp
    include/sphericalsfm/spherical_utils.h src/spherical_utils.cpp
    include/sphericalsfm/rotation_averaging.h src/rotation_averaging.cpp
    include/sphericalsfm/uncalibrated_pose_graph.h src/uncalibrated_pose_graph.cpp
)
#target_compile_features( sphericalsfm PRIVATE cxx_alias_templates )
target_link_libraries( sphericalsfm ${CERES_LIBRARIES} )
target_link_libraries( sphericalsfm ${OpenCV_LIBS} )

add_subdirectory( evaluation )
add_subdirectory( examples )

