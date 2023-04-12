find_path(PoseLib_dirs PoseLib REQUIRED)
find_library(PoseLib_libs PoseLib REQUIRED)

find_path(OpenGV_dirs opengv REQUIRED)
find_library(OpenGV_libs opengv REQUIRED)

include_directories( ${OpenGV_dirs} )
include_directories( ${PoseLib_dirs} )

add_library( five_point
    five_point_estimator.h five_point_estimator.cpp
    nister_estimator.h nister_estimator.cpp
    solver_stewenius.h solver_stewenius.cpp
    stewenius_estimator.h stewenius_estimator.cpp
    opengv_estimator.h opengv_estimator.cpp
)
target_link_libraries( five_point sphericalsfm ${PoseLib_libs} ${OpenGV_libs})

