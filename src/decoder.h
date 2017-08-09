/**
 * GLFCV - Light field disparity estimation using a guided filter cost volume
 *
 * Copyright (C) 2017 Adam Stacey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef DECODER_H
#define DECODER_H

#include <string>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>

#include "lfp.h"
#include "white-calib.h"
#include "helper.h"

struct image_meta {
  float disparity_min;
  float disparity_max;
  bool isValid;
};

namespace decoder {

class Decoder {
 public:
  Decoder(const std::string &filename, const std::string &calDataDir);

  void Decode();

  void DisplayLenslet(const std::string &window_name) const;

  void DisplayLightFieldSlices(const std::string &window_name, int refresh_speed) const;

  void WriteLensletImage(const std::string &filename, const bool raw = false) const;

  void WriteLightFieldSlices(const std::string &out_dir, const std::string &image_name) const;

  const std::vector<std::vector<cv::Mat>> &GetLightField();

 private:
  lfp::RawImage raw_data_;
  whiteCalib::WhiteImage white_data_;
  cv::Mat white_image_;
  cv::Mat raw_image_;
  cv::Mat decoded_lenslet_image_;
  std::vector<std::vector<cv::Mat>> light_field_;
};

bool ReadBenchmarkFolder(boost::filesystem::path &input_dir,
                         std::vector<std::vector<cv::Mat>> &lf,
                         cv::Mat &depth_truth, struct image_meta *meta);

bool WriteCVMatToPFM(boost::filesystem::path &save_loc, cv::Mat &image);

void DisplayCVMat(const std::string &window_name, const cv::Mat &image, double downscale);
}

#endif  // DECODER_H