# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.10)
project(nvblox_ros LANGUAGES CXX CUDA)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Default to release build
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

# CUDA architecture selection.
# If the caller already set CMAKE_CUDA_ARCHITECTURES (e.g. via colcon --cmake-args), honour it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "72;75;80;86;87")
  message(STATUS "nvblox_ros: building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "nvblox_ros: using caller-provided CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# nvblox core: build in-tree from submodule, or find pre-installed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvblox_core/CMakeLists.txt")
  message(STATUS "nvblox_ros: Building nvblox core in-tree from nvblox_core/")
  set(BUILD_PYTORCH_WRAPPER OFF CACHE BOOL "" FORCE)
  set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(USE_SYSTEM_GLOG ON CACHE BOOL "" FORCE)
  set(USE_SYSTEM_GFLAGS ON CACHE BOOL "" FORCE)
  set(USE_SYSTEM_GTEST ON CACHE BOOL "" FORCE)
  add_subdirectory(nvblox_core)
  set(NVBLOX_IN_TREE TRUE)
else()
  message(STATUS "nvblox_ros: nvblox_core/ not found, using find_package(nvblox)")
  find_package(glog REQUIRED)
  find_package(gflags REQUIRED)
  find_package(nvblox REQUIRED)
  message(STATUS "nvblox_ros: Using installed nvblox core from ${nvblox_DIR}")
  set(NVBLOX_IN_TREE FALSE)
endif()

################
# DEPENDENCIES #
################
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_auto REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(nvblox_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(libstatistics_collector REQUIRED)
find_package(message_filters REQUIRED)
find_package(Threads REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(nvblox_ros_common REQUIRED)
find_package(CUDAToolkit REQUIRED)

#############
# LIBRARIES #
#############
add_library(${PROJECT_NAME}_lib SHARED
  src/lib/conversions/image_conversions.cpp
  src/lib/conversions/image_conversions_thrust.cu
  src/lib/conversions/mesh_conversions.cpp
  src/lib/conversions/pointcloud_conversions.cu
  src/lib/conversions/esdf_slice_conversions.cu
  src/lib/conversions/esdf_and_gradients_conversions.cu
  src/lib/conversions/transform_conversions.cpp
  src/lib/layer_publishing.cpp
  src/lib/camera_cache.cpp
  src/lib/visualization.cpp
  src/lib/transformer.cpp
  src/lib/utils.cpp
  src/lib/mapper_initialization.cpp
  src/lib/nvblox_node.cpp
  src/lib/fuser_node.cpp
  src/lib/node_params.cpp
  src/lib/terminal_reading.cpp
  src/lib/rosbag_data_loader.cpp
  src/lib/rosbag_reading.cpp
)
if(NVBLOX_IN_TREE)
  set_nvblox_compiler_options(${PROJECT_NAME}_lib)
  target_link_libraries(${PROJECT_NAME}_lib PUBLIC nvblox_lib nvblox_eigen nvblox_datasets pthread glog)
  target_link_libraries(${PROJECT_NAME}_lib PRIVATE nvblox_gpu_hash nvblox_stdgpu)
else()
  target_compile_options(${PROJECT_NAME}_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
  target_link_libraries(${PROJECT_NAME}_lib PUBLIC nvblox::nvblox_lib nvblox::nvblox_eigen nvblox::nvblox_datasets pthread glog)
endif()
ament_target_dependencies(${PROJECT_NAME}_lib
  PUBLIC
  rclcpp
  rclcpp_components
  rosbag2_cpp
  sensor_msgs
  geometry_msgs
  nvblox_msgs
  std_msgs
  nav_msgs
  visualization_msgs
  std_srvs
  tf2_ros
  tf2_eigen
  message_filters
  cv_bridge
  nvblox_ros_common
)
target_include_directories(${PROJECT_NAME}_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
if(NVBLOX_IN_TREE)
  target_include_directories(${PROJECT_NAME}_lib BEFORE PRIVATE
    $<TARGET_PROPERTY:nvblox_eigen,INTERFACE_INCLUDE_DIRECTORIES>)
else()
  target_include_directories(${PROJECT_NAME}_lib BEFORE PRIVATE
    $<TARGET_PROPERTY:nvblox::nvblox_eigen,INTERFACE_INCLUDE_DIRECTORIES>)
endif()

rclcpp_components_register_nodes(${PROJECT_NAME}_lib "nvblox::NvbloxNode")
rclcpp_components_register_nodes(${PROJECT_NAME}_lib "nvblox::NvbloxHumanNode")

set_target_properties(${PROJECT_NAME}_lib PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

############
# BINARIES #
############
add_executable(nvblox_node
  src/nvblox_node_main.cpp
)
target_link_libraries(nvblox_node ${PROJECT_NAME}_lib)
set_target_properties(nvblox_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(fuser_node
  src/fuser_node_main.cpp
)
target_link_libraries(fuser_node ${PROJECT_NAME}_lib)
set_target_properties(fuser_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  BUILD_RPATH_USE_ORIGIN TRUE
  INSTALL_RPATH_USE_LINK_PATH TRUE)

###########
# INSTALL #
###########
# Install includes.
install(
  DIRECTORY include/
  DESTINATION include
)

# Install the library target.
if(NVBLOX_IN_TREE)
  install(
    TARGETS ${PROJECT_NAME}_lib nvblox_lib nvblox_eigen nvblox_datasets
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
else()
  install(
    TARGETS ${PROJECT_NAME}_lib
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )
endif()

# Install nodes.
install(
  TARGETS nvblox_node fuser_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  scripts
  DESTINATION share/${PROJECT_NAME}/scripts/
)

if(BUILD_TESTING)
  # NOTE(alexmillane): We can't use ament_lint_auto because the ignore paths are not fully respected in humble.
  find_package(ament_cmake_copyright REQUIRED)
  ament_copyright("include" "scripts" "src")
  find_package(ament_cmake_cppcheck REQUIRED)
  ament_cppcheck("include" "src")
  find_package(ament_cmake_cpplint REQUIRED)
  ament_cpplint("include" "src")
  find_package(ament_cmake_flake8 REQUIRED)
  ament_flake8("scripts")
  find_package(ament_cmake_lint_cmake REQUIRED)
  ament_lint_cmake("CMakeLists.txt")
  find_package(ament_cmake_pep257 REQUIRED)
  ament_pep257("scripts")
  find_package(ament_cmake_uncrustify REQUIRED)
  ament_uncrustify("include" "src")
  find_package(ament_cmake_xmllint REQUIRED)
  ament_xmllint()

  if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/integration_tests")
    add_subdirectory(test/integration_tests)
  endif()
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/unit_tests")
    add_subdirectory(test/unit_tests)
  endif()
endif()

##########
# EXPORT #
##########
ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}_lib
)
ament_export_targets(
  ${PROJECT_NAME}Targets
)

ament_export_dependencies(
  nvblox
  nav_msgs
  tf2_ros
  message_filters
  libstatistics_collector
  visualization_msgs
  cv_bridge
  nvblox_ros_common
  nvblox_msgs
  std_srvs
  tf2_eigen
  CUDAToolkit
)

ament_package()
