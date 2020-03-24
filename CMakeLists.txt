
cmake_minimum_required( VERSION 2.6 )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "build type" FORCE)
endif()


project( spherical-sfm )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

find_package( OpenCV REQUIRED )
include_directories( ${OPENCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
include_directories( ${EIGEN3_INCLUDE_DIR} )

find_package( OpenMVG REQUIRED )

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
    include/sphericalsfm/spherical_estimator.h src/spherical_estimator.cpp
    include/sphericalsfm/plane_estimator.h src/plane_estimator.cpp
    include/sphericalsfm/spherical_utils.h src/spherical_utils.cpp
)
target_compile_features( sphericalsfm PRIVATE cxx_alias_templates )
target_link_libraries( sphericalsfm ${CERES_LIBRARIES} )
target_link_libraries( sphericalsfm ${OpenCV_LIBS} )
target_link_libraries( sphericalsfm OpenMVG::openMVG_multiview )
target_compile_features( sphericalsfm PRIVATE cxx_alias_templates )
target_compile_options( sphericalsfm PRIVATE ${OPENMVG_CFLAGS} )

add_subdirectory( examples )

