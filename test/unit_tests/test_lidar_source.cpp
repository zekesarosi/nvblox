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

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "nvblox_ros/lidar_source.hpp"
#include "nvblox_ros/node_params.hpp"

namespace nvblox
{

namespace
{

constexpr int kCols = 64;
constexpr int kRows = 16;
constexpr float kRange = 20.f;

Lidar testLidar() {return Lidar(kCols, kRows, 0.5f, 0.3f, 0.3f);}

// An organized cloud whose point at organized index idx is the beam through
// the pixel `layout` maps idx to. A default layout is a cloud already laid out
// like the nvblox image.
sensor_msgs::msg::PointCloud2 organizedCloud(
  const Lidar & lidar, const NoReturnBeams & layout, int height, int width)
{
  sensor_msgs::msg::PointCloud2 cloud;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(static_cast<size_t>(height) * width);
  cloud.height = height;
  cloud.width = width;
  cloud.row_step = cloud.point_step * width;
  sensor_msgs::PointCloud2Iterator<float> it(cloud, "x");
  for (int idx = 0; idx < height * width; ++idx, ++it) {
    const Index2D u = layout.pixelFromIndex(idx, lidar.rows(), lidar.cols());
    Vector3f p = lidar.unprojectFromPixelIndices(u, kRange);
    if (idx % 7 == 0) {
      // Sprinkle no-returns the way an Ouster reports them.
      p = Vector3f::Constant(std::numeric_limits<float>::quiet_NaN());
    }
    it[0] = p.x();
    it[1] = p.y();
    it[2] = p.z();
  }
  return cloud;
}

NoReturnBeams layoutOf(int col_offset, bool col_reversed, bool row_reversed)
{
  NoReturnBeams layout;
  layout.col_offset = col_offset;
  layout.col_reversed = col_reversed;
  layout.row_reversed = row_reversed;
  return layout;
}

void expectFits(const NoReturnBeams & truth)
{
  const Lidar lidar = testLidar();
  const auto check = checkCloudAgainstLidar(organizedCloud(lidar, truth, kRows, kCols), lidar);
  EXPECT_TRUE(check.intrinsics_ok);
  EXPECT_TRUE(check.organized);
  EXPECT_GT(check.layout_match_fraction, 0.99f);
  EXPECT_EQ(check.layout.col_offset, truth.col_offset);
  EXPECT_EQ(check.layout.col_reversed, truth.col_reversed);
  EXPECT_EQ(check.layout.row_reversed, truth.row_reversed);
}

}  // namespace

TEST(LidarSourceTest, ImageOrderedCloudFitsIdentity) {
  expectFits(layoutOf(0, false, false));
}

TEST(LidarSourceTest, HalfTurnFrameIsFittedAsOffset) {
  // A cloud whose column 0 faces +x rather than -pi (UE bridge, or os_lidar vs
  // os_sensor) is the same grid half a revolution round.
  expectFits(layoutOf(kCols / 2, false, false));
}

TEST(LidarSourceTest, ClockwiseTopDownCloudIsFitted) {
  expectFits(layoutOf(5, true, false));
  expectFits(layoutOf(kCols / 2, true, true));
}

TEST(LidarSourceTest, OusterOsLidarLayoutIsFitted) {
  // What a real OS1-128 publishes in os_lidar: clockwise columns starting
  // just past +x (col_offset 513 of 1024), top beam first.
  expectFits(layoutOf(kCols / 2 + 1, true, false));
}

TEST(LidarSourceTest, ScrambledCloudExplainsLittle) {
  const Lidar lidar = testLidar();
  sensor_msgs::msg::PointCloud2 cloud = organizedCloud(lidar, NoReturnBeams(), kRows, kCols);
  // Swap the top quarter of rows with the third quarter: half the beams are
  // far from their index and no single orientation undoes that.
  const size_t row_bytes = cloud.row_step;
  for (int row = 0; row < kRows / 4; ++row) {
    std::swap_ranges(
      cloud.data.begin() + row * row_bytes, cloud.data.begin() + (row + 1) * row_bytes,
      cloud.data.begin() + (row + kRows / 2) * row_bytes);
  }
  const auto check = checkCloudAgainstLidar(cloud, lidar);
  EXPECT_LT(check.layout_match_fraction, 0.9f);
}

TEST(LidarSourceTest, UnorganizedCloudIsNotOrganized) {
  const Lidar lidar = testLidar();
  const auto check =
    checkCloudAgainstLidar(organizedCloud(lidar, NoReturnBeams(), 1, kRows * kCols), lidar);
  EXPECT_TRUE(check.intrinsics_ok);
  EXPECT_FALSE(check.organized);
}

TEST(LidarSourceTest, ModelThatSeesNothingFails) {
  const Lidar lidar = testLidar();
  // Every return is inside a min range longer than the cloud's range.
  const Lidar far_lidar(kCols, kRows, kRange * 2.f, 0.3f, 0.3f);
  const auto check =
    checkCloudAgainstLidar(organizedCloud(lidar, NoReturnBeams(), kRows, kCols), far_lidar);
  EXPECT_FALSE(check.intrinsics_ok);
}

LidarSourceConfig testConfig()
{
  LidarSourceConfig config;
  config.name = "rear";
  config.width = kCols;
  config.height = kRows;
  config.vertical_fov_rad = 0.6f;
  config.min_valid_range_m = 0.5f;
  return config;
}

TEST(LidarSourceTest, ValidationWithoutDepthDisablesCarving) {
  LidarSource source(testConfig());
  NoReturnBeams layout = layoutOf(kCols / 2, false, false);
  source.markValidated(layout);
  EXPECT_FALSE(source.miss_ray_allowed());
  layout.depth_m = 160.f;
  source.markValidated(layout);
  EXPECT_TRUE(source.miss_ray_allowed());
  EXPECT_EQ(source.no_return_beams().col_offset, kCols / 2);
}

TEST(LidarSourceTest, RejectionIsRetriedNotLatched) {
  LidarSource source(testConfig());
  const rclcpp::Time t0(100, 0, RCL_ROS_TIME);
  EXPECT_TRUE(source.validationDue(t0));
  source.markRejected(t0);
  EXPECT_TRUE(source.rejected());
  EXPECT_FALSE(source.miss_ray_allowed());
  EXPECT_FALSE(source.validationDue(t0 + rclcpp::Duration::from_seconds(0.5)));
  EXPECT_TRUE(source.validationDue(t0 + rclcpp::Duration::from_seconds(1.0)));
  // A bag loop or sim reset moves time backwards: retry rather than wait.
  EXPECT_TRUE(source.validationDue(rclcpp::Time(50, 0, RCL_ROS_TIME)));
  NoReturnBeams layout;
  layout.depth_m = 160.f;
  source.markValidated(layout);
  EXPECT_FALSE(source.validationDue(t0 + rclcpp::Duration::from_seconds(10.0)));
  EXPECT_FALSE(source.rejected());
}

class LidarSourceParamsTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

TEST_F(LidarSourceParamsTest, OverridesFallBackToNodeWideParams) {
  rclcpp::NodeOptions options;
  options.parameter_overrides(
  {
    rclcpp::Parameter("lidar_width", 1024),
    rclcpp::Parameter("lidar_height", 128),
    rclcpp::Parameter("integrate_lidar_rate_hz", 20.0),
    rclcpp::Parameter("lidars.front.topic", "/front/ouster/points"),
    // YAML often writes whole numbers as integers.
    rclcpp::Parameter("lidars.front.min_valid_range_m", 3),
    rclcpp::Parameter("lidars.front.miss_ray_carving", false),
  });
  auto node = std::make_shared<rclcpp::Node>("lidar_source_params_test", options);
  NvbloxNodeParams params;
  initializeNvbloxNodeParams(node.get(), &params);

  const LidarSourceConfig front = declareLidarSourceConfig(node.get(), "front", params);
  const LidarSourceConfig rear = declareLidarSourceConfig(node.get(), "rear", params);

  EXPECT_EQ(front.topic, "/front/ouster/points");
  EXPECT_EQ(front.width, 1024);
  EXPECT_EQ(front.height, 128);
  EXPECT_FLOAT_EQ(front.min_valid_range_m, 3.0f);
  EXPECT_FLOAT_EQ(front.integrate_rate_hz, 20.0f);
  EXPECT_FALSE(front.miss_ray_carving);

  EXPECT_EQ(rear.topic, "rear");
  EXPECT_EQ(rear.width, 1024);
  EXPECT_FLOAT_EQ(rear.min_valid_range_m, params.lidar_min_valid_range_m.get());
  EXPECT_TRUE(rear.miss_ray_carving);

  EXPECT_THROW(declareLidarSourceConfig(node.get(), "front.width", params), std::invalid_argument);

  // Two sources with identical models are still separate sources.
  LidarSource a(front);
  LidarSource b(rear);
  EXPECT_NE(a.queue_name(), b.queue_name());
  EXPECT_EQ(a.debugTopic("depth_image"), "~/lidars/front/depth_image");
}

}  // namespace nvblox
