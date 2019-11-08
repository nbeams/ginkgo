add_executable(luca_test_solver simple-solver.cpp)
target_link_libraries(luca_test_solver ginkgo)
target_include_directories(luca_test_solver PRIVATE ${PROJECT_SOURCE_DIR})
