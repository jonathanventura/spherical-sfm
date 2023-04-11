include_directories( . )
add_subdirectory( problem_generator )

find_path(PoseLib_dirs PoseLib REQUIRED)
find_library(PoseLib_libs PoseLib REQUIRED)

include_directories( ${PoseLib_dirs} )

add_executable(test_random_problems 
    five_point_estimator.h five_point_estimator.cpp
    nister_estimator.h nister_estimator.cpp
    solver_stewenius.h solver_stewenius.cpp
    stewenius_estimator.h stewenius_estimator.cpp
    test_random_problems.cpp)
target_link_libraries(test_random_problems sphericalsfm gflags problem_generator ${PoseLib_libs})

