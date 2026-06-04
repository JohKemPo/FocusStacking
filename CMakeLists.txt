cmake_minimum_required(VERSION 3.20)
project(FocusStackingCPP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configurações de Otimização
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -Wall")

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(CURL REQUIRED)
include_directories(include)

file(GLOB_RECURSE SRC_FILES src/*.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    opencv_core opencv_imgproc opencv_imgcodecs opencv_video opencv_calib3d opencv_features2d
    spdlog::spdlog 
    yaml-cpp
    CURL::libcurl
)