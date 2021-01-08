# Set link libraries
set(TEST_LINK_LIBS
    argparse::argparse
    ${OpenCV_LIBS}
)



# Kalman filter test
set(KALMAN_TEST_SRCS
    kalman_filter_test.cpp
    ../src/tracker/kalman_filter.cpp
)

# Set include directories
set(KALMAN_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

# Build KalmanFilter test
add_executable(kalman_filter_test ${KALMAN_TEST_SRCS})

# Link dependencies
target_link_libraries(kalman_filter_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(kalman_filter_test PRIVATE ${KALMAN_TEST_INC_DIRS})


# ViBe test
set(VIBE_SRCS
    vibe_test.cpp
    ../src/bgsegm/vibe.cpp
    ../src/bgsegm/vibe_sequential.cpp
)

# Set include directories
set(VIBE_INC_DIRS
    ../src/bgsegm
    ${OpenCV_INCLUDE_DIRS}
)

# Build ViBe test
add_executable(vibe_test ${VIBE_SRCS})

# Link dependencies
target_link_libraries(vibe_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(vibe_test PRIVATE ${VIBE_INC_DIRS})



# LAP Solver test
set(LAP_SOLVER_TEST_SRCS
    lap_solver_test.cpp
    ../src/tracker/lap_solver.cpp
)

# Set include directories
set(LAP_SOLVER_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

# Build LAPSolver test
add_executable(lap_solver_test ${LAP_SOLVER_TEST_SRCS})

# Link dependencies
target_link_libraries(lap_solver_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(lap_solver_test PRIVATE ${LAP_SOLVER_TEST_INC_DIRS})


# SORT Tracker test
set(SORT_TRACKER_TEST_SRCS
    sort_tracker_test.cpp
    ../src/tracker/lap_solver.cpp
    ../src/tracker/tracked_bbox.cpp
    ../src/tracker/tracker.cpp
    ../src/tracker/kalman_filter.cpp
)

# Set include directories
set(SORT_TRACKER_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

# Build LAPSolver test
add_executable(sort_tracker_test ${SORT_TRACKER_TEST_SRCS})

# Link dependencies
target_link_libraries(sort_tracker_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(sort_tracker_test PRIVATE ${SORT_TRACKER_TEST_INC_DIRS})
