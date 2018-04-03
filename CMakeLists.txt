cmake_minimum_required(VERSION 3.5)

project(B2FRender)

set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".build/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ".build/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ".build/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wreturn-type")

set(SRC b2f_render.cpp mesh.hpp geometry.hpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(B2FRenderExec ${SRC})
