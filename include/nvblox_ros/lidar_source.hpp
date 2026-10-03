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

#ifndef NVBLOX_ROS__LIDAR_SOURCE_HPP_
#define NVBLOX_ROS__LIDAR_SOURCE_HPP_

#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/no_return_beams.h"

namespace nvblox
{

class NvbloxNodeParams;

/// One LiDAR feeding the map. Fields default to the node-wide lidar_* params.
struct LidarSourceConfig
{
  std::string name;
  std::string topic;

  int width = 0;
  int height = 0;
  bool use_non_equal_vertical_fov = false;
  float vertical_fov_rad = 0.f;
  float min_angle_below_zero_elevation_rad = 0.f;
  float max_angle_above_zero_elevation_rad = 0.f;
  float min_valid_range_m = 0.f;

  float integrate_rate_hz = 0.f;
  bool use_motion_compensation = false;
  // Needs lidar_no_return_free_depth_m > 0 and a verified layout.
  bool miss_ray_carving = true;

  Lidar makeLidar() const;
};

/// Reads `lidars.<name>.*`, falling back to the node-wide lidar params.
LidarSourceConfig declareLidarSourceConfig(
  rclcpp::Node * node, const std::string & name, const NvbloxNodeParams & params);

/// Outcome of checking one received cloud against a source's sensor model.
struct LidarCloudCheck
{
  bool intrinsics_ok = false;
  bool organized = false;
  // Best-fitting index -> pixel layout (depth unset).
  NoReturnBeams layout;
  // Fraction of projected returns `layout` explains.
  float layout_match_fraction = 0.f;
  int num_valid_points = 0;
  int num_projected_points = 0;
};

/// Checks a cloud against `lidar` on the CPU. For validation, not every scan.
LidarCloudCheck checkCloudAgainstLidar(
  const sensor_msgs::msg::PointCloud2 & cloud, const Lidar & lidar);

/// Per-source counters. Atomics are written by the subscription callback.
struct LidarSourceStats
{
  std::atomic<uint64_t> received{0};
  std::atomic<uint64_t> dropped_queue{0};
  std::atomic<uint64_t> rejected_stamp{0};
  uint64_t integrated = 0;
  uint64_t skipped_rate = 0;
  uint64_t skipped_shape_clearing = 0;
  uint64_t skipped_tf = 0;
  uint64_t rejected_model = 0;
  uint64_t motion_compensation_fallbacks = 0;
};

/// One LiDAR feeding the shared map, with its own queue and rate limiter.
class LidarSource
{
public:
  using PointcloudMsg = sensor_msgs::msg::PointCloud2::ConstSharedPtr;

  explicit LidarSource(LidarSourceConfig config);

  const LidarSourceConfig & config() const {return config_;}
  const std::string & name() const {return config_.name;}
  const Lidar & lidar() const {return lidar_;}
  const std::string & queue_name() const {return queue_name_;}

  std::unique_ptr<std::list<PointcloudMsg>> & queue() {return queue_;}
  std::mutex * queue_mutex() {return &queue_mutex_;}

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_image_publisher;

  /// Pending sources integrate without carving; rejected ones are re-checked.
  enum class Validation {kPending, kRejected, kValidated};
  Validation validation() const {return validation_;}
  bool validated() const {return validation_ == Validation::kValidated;}
  bool rejected() const {return validation_ == Validation::kRejected;}
  /// Throttled: validation walks every point on the CPU.
  bool validationDue(const rclcpp::Time & now) const;
  void markPending(const rclcpp::Time & now);
  void markRejected(const rclcpp::Time & now);
  /// `no_return_beams` with depth 0 disables carving for this source.
  void markValidated(const NoReturnBeams & no_return_beams);

  bool miss_ray_allowed() const {return no_return_beams_.enabled();}
  const NoReturnBeams & no_return_beams() const {return no_return_beams_;}

  rclcpp::Time last_integrated_time = rclcpp::Time(0, 0, RCL_ROS_TIME);

  LidarSourceStats stats;

  std::string debugTopic(const std::string & leaf) const;

  const std::string & rate_key() const {return rate_key_;}
  const std::string & delay_key() const {return delay_key_;}

private:
  LidarSourceConfig config_;
  Lidar lidar_;
  std::string queue_name_;
  std::string rate_key_;
  std::string delay_key_;

  std::unique_ptr<std::list<PointcloudMsg>> queue_;
  std::mutex queue_mutex_;

  Validation validation_ = Validation::kPending;
  std::optional<rclcpp::Time> last_validation_attempt_;
  NoReturnBeams no_return_beams_;
};

std::string toString(const LidarSourceConfig & config);
std::string toString(const LidarSource & source);

}  // namespace nvblox

#endif  // NVBLOX_ROS__LIDAR_SOURCE_HPP_
