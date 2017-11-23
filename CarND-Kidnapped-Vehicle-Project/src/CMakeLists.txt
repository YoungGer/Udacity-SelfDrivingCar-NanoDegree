cmake_minimum_required(VERSION 3.7)
project(src)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        helper_functions.h
        json.hpp
        main.cpp
        map.h
        particle_filter.cpp
        particle_filter.h)

add_executable(src ${SOURCE_FILES})