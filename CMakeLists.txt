cmake_minimum_required(VERSION 2.8)
project(orb_extract_match)

set(CMAKE_BUILD_TYPE "Debug")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_FLAGS "-std=c++11 ${SSE_FLAGS} -g -O3 -march=native")

find_package(OpenCV 3 REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
)

add_library(orb_extract_match SHARED
    orb_extractor/orb_extractor.cpp
    orb_matcher/orb_matcher.cpp
)

add_executable(example example.cpp)
target_link_libraries(example ${OpenCV_LIBS} orb_extract_match)

