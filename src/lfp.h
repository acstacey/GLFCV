/**
 * LF-TADCG-CUDA-DISP - Code for estimating the disparity map of a light field
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

#ifndef LFP_H
#define LFP_H

#include <string>
#include <vector>

#include "json11.hpp"

namespace lfp {

std::vector<uint16_t> decode10bitData(const std::vector<uint8_t> &data,
                                      bool little_endian);

struct Section {
  Section(const std::vector<uint8_t> &buffer, size_t &index);
  std::string typecode;
  std::string sha1;
  std::string name;
  std::vector<uint8_t> data;
  void Identify(const Section &table);
};

struct RawImage {
  RawImage(const Section &raw_image, const Section &metadata);
  size_t width, height;
  double lensPitch, pixelPitch, rotation, xMlaScale, yMlaScale, exitPupilOffsetZ;
  double mlaSensorOffsetX, mlaSensorOffsetY, mlaSensorOffsetZ;
  int zoomStep, focusStep;
  std::string bayer_pattern;
  uint32_t min_val, max_val;
  std::vector<uint16_t> image_data;
};

struct File {
  File(const std::string &filename);
  std::string filename;
  std::vector<Section> sections;
  RawImage GetRawImage() const;
};
};

#endif  // LFP_H