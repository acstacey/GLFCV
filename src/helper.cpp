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

#include "helper.h"
#include <opencv2/core/cuda.hpp>

void MatToCSV(const cv::Mat &m, const std::string &filename) {
  std::ofstream myfile;
  myfile.open(filename.c_str());
  myfile << cv::format(m, cv::Formatter::FMT_CSV) << std::endl;
  myfile.close();
}

void MatToCSV(const cv::cuda::GpuMat &m, const std::string &filename) {
  cv::Mat temp;
  m.download(temp);
  MatToCSV(temp, filename);
}

double TriangleArea(const cv::Point2d &a, const cv::Point2d &b,
                    const cv::Point2d &c) {
  return std::abs(a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)) /
      2.0;
}

bool IsMachineBigEndian() {
  unsigned int x = 1;
  char *c = (char *) &x;
  return (int) *c == 0;
}
