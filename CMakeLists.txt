cmake_minimum_required(VERSION 3.15)

# Set project name
project(falling_object_detection)

# Set c++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find necessary dependencies
find_package(OpenCV REQUIRED)
find_package(argparse REQUIRED)
include(${CMAKE_HOME_DIRECTORY}/cmake/findFFmpeg.cmake)

# Set link libraries
set(PROJ_LINK_LIBS
    argparse::argparse
    ${FFMPEG_LIBRARIES}
    ${OpenCV_LIBS}
)

# Set include directories
set(PROJ_INC_DIRS
    src
    src/codec
    src/bgsegm
    src/kalman_filter
    src/tracker
    ${FFMPEG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)


# When cross-compile for rockchip platform, find special dependencies
if (ROCKCHIP_PLATFORM)
    find_package(PkgConfig REQUIRED)
    pkg_search_module(RGA REQUIRED librga)

    list(APPEND PROJ_LINK_LIBS ${RGA_LIBRARIES})
    list(APPEND PROJ_INC_DIRS ${RGA_INCLUDE_DIRS})

    add_compile_definitions(ROCKCHIP_PLATFORM)
    
else()
    list(APPEND PROJ_LINK_LIBS ${SWSCALE_LIBRARIES})
    list(APPEND PROJ_INC_DIRS ${SWSCALE_INCLUDE_DIRS})
endif()

# Enumerate source files
set(PROJ_SRCS
    src/main.cpp
    src/utils.cpp
    src/codec/decoder.cpp
    src/codec/video_reader.cpp
    src/bgsegm/vibe_sequential.cpp
    src/tracker/kalman_filter.cpp
    src/tracker/tracker.cpp
    src/tracker/tracked_bbox.cpp
    src/tracker/trajectory.cpp
    src/tracker/lap_solver.cpp
)

# Build executable
add_executable(${CMAKE_PROJECT_NAME} ${PROJ_SRCS})

# Link dependencies
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${PROJ_LINK_LIBS})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PROJ_INC_DIRS})

add_subdirectory(test)