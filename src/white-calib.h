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

#ifndef LIGHTFIELDOPENCV_WHITECALIB_H
#define LIGHTFIELDOPENCV_WHITECALIB_H

#include <string>
#include <vector>

#include "json11.hpp"

namespace whiteCalib {

const std::string DEFAULT_MLA_CALIB_FILE = "mla_calibration.json";
const std::string DEFAULT_FILE_PREFIX = "MOD_";
const std::string DEFAULT_META_FILETYPE = ".TXT";
const std::string DEFAULT_IMAGE_FILETYPE = ".RAW";
const std::string DEFAULT_TOOLBOX_FILETYPE = ".grid.json";

struct LensletGridModel {
  LensletGridModel();
  double exitPupilDistanceMilli;
  double xMlaSensorOffsetMicrons;
  double yMlaSensorOffsetMicrons;
  double xMlaScale;
  double yMlaScale;
  double xDiskOriginPixels;
  double yDiskOriginPixels;
  double xDiskStepPixelsX;
  double xDiskStepPixelsY;
  double yDiskStepPixelsX;
  double yDiskStepPixelsY;
  double rotation;
  bool horizOrient;
};

struct WhiteImage {
  WhiteImage(const std::string &calDataDir, int desired_zoom_step,
             int desired_foc_step);
  size_t image_index;
  size_t width, height;
  uint32_t min_val, max_val;
  std::vector<uint16_t> image_data;
  int zoomStep, focusStep;
  double exposureDuration;
  LensletGridModel mla_grid_model;
  bool used_lftoolbox_calib;
};
};

#endif  // LIGHTFIELDOPENCV_WHITECALIB_H
