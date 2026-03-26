// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef NVBLOX_ROS__CONVERSIONS__OCCUPANCY_GRID_3D_CONVERSIONS_HPP_
#define NVBLOX_ROS__CONVERSIONS__OCCUPANCY_GRID_3D_CONVERSIONS_HPP_

#include <nvblox/nvblox.h>

#include <std_msgs/msg/float32_multi_array.hpp>

namespace nvblox
{
namespace conversions
{

struct OccupancyGrid3DStats
{
  size_t total_voxels = 0;
  size_t observed_voxels = 0;
  size_t occupied_voxels = 0;
  size_t free_voxels = 0;
  size_t unobserved_voxels = 0;
  float min_log_odds = 0.0f;
  float max_log_odds = 0.0f;
};

class OccupancyGrid3DConverter
{
public:
  OccupancyGrid3DConverter()
  : gpu_grid_(MemoryType::kDevice), cpu_grid_(MemoryType::kHost) {}
  ~OccupancyGrid3DConverter() = default;

  /// Extracts log-odds from the OccupancyLayer within an AABB to a dense
  /// Float32MultiArray. Unobserved voxels (in unallocated blocks) are filled
  /// with the unobserved_value (NaN by default).
  ///
  /// The output message layout matches the ESDF 3D grid format:
  ///   data_offset = 4 (voxel_size, origin_x, origin_y, origin_z)
  ///   layout.dim[0..2] = nx, ny, nz
  ///   data[4..] = 1 float per voxel (log-odds)
  std_msgs::msg::Float32MultiArray occupancyInAabbToMultiArrayMsg(
    const OccupancyLayer & occ_layer,
    const AxisAlignedBoundingBox & aabb,
    float unobserved_value,
    const CudaStream & cuda_stream);

  /// After calling occupancyInAabbToMultiArrayMsg, retrieve statistics
  /// computed during the last conversion (computed on CPU from the result grid).
  const OccupancyGrid3DStats & lastStats() const {return last_stats_;}

private:
  void computeStats(
    const std::vector<float> & data, size_t num_voxels,
    float unobserved_value);

  Unified3DGrid<float> gpu_grid_;
  Unified3DGrid<float> cpu_grid_;
  OccupancyGrid3DStats last_stats_;
};

}  // namespace conversions
}  // namespace nvblox

#endif  // NVBLOX_ROS__CONVERSIONS__OCCUPANCY_GRID_3D_CONVERSIONS_HPP_
