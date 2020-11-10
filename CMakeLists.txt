cmake_minimum_required(VERSION 2.8)
project(
    "ComputerVision"
    LANGUAGES CXX
)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(src/image_processing)
add_subdirectory(src/features)
