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

set(KALMAN_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(kalman_filter_test ${KALMAN_TEST_SRCS})
target_link_libraries(kalman_filter_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(kalman_filter_test PRIVATE ${KALMAN_TEST_INC_DIRS})


# ViBe test
set(VIBE_SRCS
    vibe_test.cpp
    ../src/bgsegm/vibe.cpp
    ../src/bgsegm/vibe_sequential.cpp
)

set(VIBE_INC_DIRS
    ../src/bgsegm
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(vibe_test ${VIBE_SRCS})
target_link_libraries(vibe_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(vibe_test PRIVATE ${VIBE_INC_DIRS})


# LAP Solver test
set(LAP_SOLVER_TEST_SRCS
    lap_solver_test.cpp
    ../src/tracker/lap_solver.cpp
)

set(LAP_SOLVER_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(lap_solver_test ${LAP_SOLVER_TEST_SRCS})
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

set(SORT_TRACKER_TEST_INC_DIRS
    ../src/tracker
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(sort_tracker_test ${SORT_TRACKER_TEST_SRCS})
target_link_libraries(sort_tracker_test PRIVATE ${TEST_LINK_LIBS})
target_include_directories(sort_tracker_test PRIVATE ${SORT_TRACKER_TEST_INC_DIRS})

# VideoReader test
set(VIDEO_READER_TEST_SRCS
    video_reader_test.cpp
    ../src/codec/video_reader.cpp
)
add_executable(video_reader_test ${VIDEO_READER_TEST_SRCS})

set(VIDEO_READER_TEST_INC_DIRS
    ../src/codec
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

set(VIDEO_READER_TEST_LINK_LIBS
    ${TEST_LINK_LIBS} 
    ${FFMPEG_LIBRARIES}
)

if (ROCKCHIP_PLATFORM)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(RGA REQUIRED librga)
    pkg_search_module(DRM REGUIRED libdrm)

    list(APPEND VIDEO_READER_TEST_LINK_LIBS ${RGA_LIBRARIES})
    list(APPEND VIDEO_READER_TEST_INC_DIRS ${RGA_INCLUDE_DIRS} ${DRM_INCLUDE_DIRS})

    target_compile_definitions(video_reader_test PRIVATE ROCKCHIP_PLATFORM)
else()
    list(APPEND VIDEO_READER_TEST_LINK_LIBS ${SWSCALE_LIBRARIES})
    list(APPEND VIDEO_READER_TEST_INC_DIRS ${SWSCALE_INCLUDE_DIRS})
endif()

target_link_libraries(video_reader_test PRIVATE ${VIDEO_READER_TEST_LINK_LIBS})
target_include_directories(video_reader_test PRIVATE ${VIDEO_READER_TEST_INC_DIRS})
