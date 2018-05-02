cmake_minimum_required(VERSION 3.5)

project(B2FRender)

set(CMAKE_CXX_COMPILER clang++)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wreturn-type")


FILE(GLOB FILES
        ./*.hpp
        ./*.cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(B2FRenderExec ${FILES})
