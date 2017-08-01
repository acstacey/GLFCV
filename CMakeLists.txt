###############################################################################
# LF-TADCG-CUDA-DISP - Code for estimating the disparity map of a light field
#
# Copyright (C) 2017 Adam Stacey
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################

#Change this if you need to target a specific CMake version
cmake_minimum_required(VERSION 2.8)

# Enable release by default
#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build (Debug or Release)" FORCE)
#endif()

# Enable C++14
set(CMAKE_CXX_STANDARD 14)

# Set up project
project(LightFieldCudaDepth)

# Detect and add OpenCV
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
find_package(OpenCV REQUIRED HINTS /usr/local/opt/opencv3)

find_package(CUDA 8.0 REQUIRED)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.58 REQUIRED COMPONENTS filesystem regex)

# Add files
file(GLOB SOURCES src/*.cpp src/cuda-kernels/*.cu)
file(GLOB HEADERS src/*.h src/cuda-kernels*.h)
source_group("Header Files" FILES ${HEADERS})


# Define sources and executable
set(EXECUTABLE_NAME "LightFieldDepth")
cuda_add_executable(${EXECUTABLE_NAME} ${SOURCES} ${HEADERS})

# Link against OpenCV
include_directories(${OpenCV_INCLUDE_DIR})
target_link_libraries(${EXECUTABLE_NAME} ${OpenCV_LIBS})

# Link against CUDA
include_directories(${CUDA_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} ${CUDA_LIBRARIES})

# Link against boost filesystem
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(${EXECUTABLE_NAME} ${Boost_LIBRARIES})
