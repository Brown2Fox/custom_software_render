cmake_minimum_required(VERSION 3.5)

project(B2FRender)

#set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../../out_render/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../../out_render/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "../../out_render/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SRC b2f_render.cpp Mesh.cpp Mesh.h b2f_render_jni.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories("C:/Program Files/Java/jdk-9.0.1/include")
include_directories("C:/Program Files/Java/jdk-9.0.1/include/win32")
include_directories(".")
#include_directories("$ENV{MSYS_DIR}/mingw64/lib/gcc/x86_64-w64-mingw32/7.2.0/include")

add_executable(B2FRenderExec ${SRC})
add_library(B2FRenderLib SHARED ${SRC})