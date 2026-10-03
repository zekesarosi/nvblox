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

#include "nvblox_ros/lidar_source.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "nvblox_ros/node_params.hpp"
#include "nvblox_ros/utils.hpp"

namespace nvblox
{

namespace
{

template<typename T>
T declareOverride(
  rclcpp::Node * node, const std::string & name, const std::string & leaf,
  const T & fallback, const std::string & help)
{
  rcl_interfaces::msg::ParameterDescriptor desc;
  desc.description = help;
  return node->declare_parameter<T>("lidars." + name + "." + leaf, fallback, desc);
}

// YAML `4` is an integer; a double-typed declaration would throw on it.
double declareNumberOverride(
  rclcpp::Node * node, const std::string & name, const std::string & leaf,
  double fallback, const std::string & help)
{
  rcl_interfaces::msg::ParameterDescriptor desc;
  desc.description = help;
  desc.dynamic_typing = true;
  const std::string key = "lidars." + name + "." + leaf;
  const rclcpp::ParameterValue value =
    node->declare_parameter(key, rclcpp::ParameterValue(fallback), desc);
  switch (value.get_type()) {
    case rclcpp::ParameterType::PARAMETER_DOUBLE:
      return value.get<double>();
    case rclcpp::ParameterType::PARAMETER_INTEGER:
      return static_cast<double>(value.get<int64_t>());
    default:
      throw std::invalid_argument(key + " must be a number");
  }
}

bool isPlainIdentifier(const std::string & name)
{
  if (name.empty()) {
    return false;
  }
  for (const char c : name) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }
  return true;
}

constexpr double kValidationRetryPeriodS = 1.0;

// Real OS1-128 returns land up to 2 rows from their index (uneven beams).
constexpr int kLayoutRowTolerance = 3;
constexpr int kLayoutColTolerance = 2;

}  // namespace

Lidar LidarSourceConfig::makeLidar() const
{
  if (use_non_equal_vertical_fov) {
    return Lidar(
      width, height, min_valid_range_m, min_angle_below_zero_elevation_rad,
      max_angle_above_zero_elevation_rad);
  }
  return Lidar(width, height, min_valid_range_m, vertical_fov_rad);
}

LidarSourceConfig declareLidarSourceConfig(
  rclcpp::Node * node, const std::string & name, const NvbloxNodeParams & params)
{
  if (!isPlainIdentifier(name)) {
    throw std::invalid_argument(
            "lidar_names entry '" + name + "' must be letters, digits and underscores");
  }
  LidarSourceConfig c;
  c.name = name;
  c.topic = declareOverride<std::string>(
    node, name, "topic", name,
    "PointCloud2 topic for this LiDAR. Relative names resolve in the node namespace.");
  c.width = declareOverride<int>(
    node, name, "width", params.lidar_width, "Azimuth divisions (columns).");
  c.height = declareOverride<int>(
    node, name, "height", params.lidar_height, "Elevation divisions (beams / rows).");
  c.use_non_equal_vertical_fov = declareOverride<bool>(
    node, name, "use_non_equal_vertical_fov", params.use_non_equal_vertical_fov_lidar_params,
    "Use the asymmetric below/above elevation limits instead of vertical_fov_rad.");
  c.vertical_fov_rad = declareNumberOverride(
    node, name, "vertical_fov_rad", params.lidar_vertical_fov_rad,
    "Symmetric vertical field of view.");
  c.min_angle_below_zero_elevation_rad = declareNumberOverride(
    node, name, "min_angle_below_zero_elevation_rad",
    params.min_angle_below_zero_elevation_rad, "Lowest beam below the sensor horizon.");
  c.max_angle_above_zero_elevation_rad = declareNumberOverride(
    node, name, "max_angle_above_zero_elevation_rad",
    params.max_angle_above_zero_elevation_rad, "Highest beam above the sensor horizon.");
  c.min_valid_range_m = declareNumberOverride(
    node, name, "min_valid_range_m", params.lidar_min_valid_range_m,
    "Returns closer than this are dropped (the airframe self-filter).");
  c.integrate_rate_hz = declareNumberOverride(
    node, name, "integrate_rate_hz", params.integrate_lidar_rate_hz,
    "Maximum integration rate for this LiDAR, independent of the others.");
  c.use_motion_compensation = declareOverride<bool>(
    node, name, "use_motion_compensation", params.use_lidar_motion_compensation,
    "Motion-compensate scans from per-point timestamps.");
  c.miss_ray_carving = declareOverride<bool>(
    node, name, "miss_ray_carving", true,
    "Carve free space along no-return beams (needs lidar_no_return_free_depth_m > 0).");
  return c;
}

LidarCloudCheck checkCloudAgainstLidar(
  const sensor_msgs::msg::PointCloud2 & cloud, const Lidar & lidar)
{
  LidarCloudCheck check;
  const int rows = lidar.num_elevation_divisions();
  const int cols = lidar.num_azimuth_divisions();
  check.organized = static_cast<int>(cloud.height) == rows &&
    static_cast<int>(cloud.width) == cols;

  std::vector<std::pair<int, Index2D>> projected;
  projected.reserve(static_cast<size_t>(cloud.width) * cloud.height);
  sensor_msgs::PointCloud2ConstIterator<float> it(cloud, "x");
  for (int idx = 0; it != it.end(); ++it, ++idx) {
    const Vector3f p(it[0], it[1], it[2]);
    if (p.hasNaN() || !lidar.isInValidRange(p)) {
      continue;
    }
    ++check.num_valid_points;
    Index2D u;
    if (lidar.project(p, &u)) {
      projected.emplace_back(idx, u);
    }
  }
  check.intrinsics_ok = !projected.empty();
  check.num_projected_points = static_cast<int>(projected.size());
  if (!check.organized || projected.empty()) {
    return check;
  }

  // Keep the orientation and column offset that explain the most returns.
  for (const bool col_reversed : {false, true}) {
    for (const bool row_reversed : {false, true}) {
      NoReturnBeams candidate;
      candidate.col_reversed = col_reversed;
      candidate.row_reversed = row_reversed;
      std::vector<int> offset_counts(cols, 0);
      for (const auto & [idx, u] : projected) {
        const Index2D mapped = candidate.pixelFromIndex(idx, rows, cols);
        ++offset_counts[((u.x() - mapped.x()) % cols + cols) % cols];
      }
      candidate.col_offset = static_cast<int>(
        std::max_element(offset_counts.begin(), offset_counts.end()) -
        offset_counts.begin());

      int num_match = 0;
      for (const auto & [idx, u] : projected) {
        const Index2D mapped = candidate.pixelFromIndex(idx, rows, cols);
        const int col_diff = ((u.x() - mapped.x()) % cols + cols) % cols;
        const int col_err = std::min(col_diff, cols - col_diff);
        if (std::abs(u.y() - mapped.y()) <= kLayoutRowTolerance &&
          col_err <= kLayoutColTolerance)
        {
          ++num_match;
        }
      }
      const float fraction = static_cast<float>(num_match) / projected.size();
      if (fraction > check.layout_match_fraction) {
        check.layout_match_fraction = fraction;
        check.layout = candidate;
      }
    }
  }
  return check;
}

LidarSource::LidarSource(LidarSourceConfig config)
: config_(std::move(config)),
  lidar_(config_.makeLidar()),
  queue_name_("pointcloud_queue/" + config_.name),
  rate_key_("ros/lidar/" + config_.name),
  delay_key_("ros/pointcloud_integration/" + config_.name),
  queue_(std::make_unique<std::list<PointcloudMsg>>())
{
}

bool LidarSource::validationDue(const rclcpp::Time & now) const
{
  if (validated()) {
    return false;
  }
  return !last_validation_attempt_ ||
         (now - *last_validation_attempt_).seconds() >= kValidationRetryPeriodS ||
         now < *last_validation_attempt_;
}

void LidarSource::markPending(const rclcpp::Time & now)
{
  validation_ = Validation::kPending;
  last_validation_attempt_ = now;
  no_return_beams_ = NoReturnBeams();
}

void LidarSource::markRejected(const rclcpp::Time & now)
{
  validation_ = Validation::kRejected;
  last_validation_attempt_ = now;
  no_return_beams_ = NoReturnBeams();
}

void LidarSource::markValidated(const NoReturnBeams & no_return_beams)
{
  validation_ = Validation::kValidated;
  no_return_beams_ = no_return_beams;
}

std::string LidarSource::debugTopic(const std::string & leaf) const
{
  return "~/lidars/" + makeSafeTopicName(config_.name) + "/" + leaf;
}

std::string toString(const LidarSourceConfig & c)
{
  std::ostringstream ss;
  ss << c.name << ": topic=" << c.topic << " " << c.width << "x" << c.height;
  if (c.use_non_equal_vertical_fov) {
    ss << " elev=[-" << std::fabs(c.min_angle_below_zero_elevation_rad) << ", +" <<
      c.max_angle_above_zero_elevation_rad << "] rad";
  } else {
    ss << " vfov=" << c.vertical_fov_rad << " rad";
  }
  ss << " min_range=" << c.min_valid_range_m << " m rate=" << c.integrate_rate_hz <<
    " Hz mc=" << (c.use_motion_compensation ? "on" : "off") <<
    " miss_ray=" << (c.miss_ray_carving ? "on" : "off");
  return ss.str();
}

std::string toString(const LidarSource & s)
{
  std::ostringstream ss;
  ss << s.name() << ": received=" << s.stats.received.load() << " integrated=" <<
    s.stats.integrated << " skipped(rate=" << s.stats.skipped_rate << " shape_clear=" <<
    s.stats.skipped_shape_clearing << " tf=" << s.stats.skipped_tf << " model=" <<
    s.stats.rejected_model << " stamp=" << s.stats.rejected_stamp.load() <<
    ") dropped_queue=" << s.stats.dropped_queue.load() << " mc_fallbacks=" <<
    s.stats.motion_compensation_fallbacks << " validation=" <<
    (s.validated() ? "ok" : s.rejected() ? "REJECTED" : "pending") <<
    " miss_ray=" << (s.miss_ray_allowed() ? "on" : "off");
  return ss.str();
}

}  // namespace nvblox
