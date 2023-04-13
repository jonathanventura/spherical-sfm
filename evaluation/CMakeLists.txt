include_directories( . )
add_subdirectory( problem_generator )
add_subdirectory( five_point )

add_executable(test_random_problems test_random_problems.cpp)
target_link_libraries(test_random_problems gflags problem_generator five_point)

add_executable(test_ransac vanilla_ransac.h test_ransac.cpp)
target_link_libraries(test_ransac gflags problem_generator five_point )

