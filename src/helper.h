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

#ifndef LIGHTFIELDOPENCV_HELPER_H
#define LIGHTFIELDOPENCV_HELPER_H

#include <fstream>
#include <opencv2/core/core.hpp>
#include <chrono>
#include <type_traits>
#include <ctime>

#define CPU_TIME() (((double) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) / 1.0e6)
//#define MIN(a,b) (((a)<(b))?(a):(b))
//#define MAX(a,b) (((a)>(b))?(a):(b))

void MatToCSV(const cv::Mat &m, const std::string &filename);
void MatToCSV(const cv::cuda::GpuMat &m, const std::string &filename);
double TriangleArea(const cv::Point2d &a, const cv::Point2d &b,
                    const cv::Point2d &c);
bool IsMachineBigEndian();

#endif //LIGHTFIELDOPENCV_HELPER_H
