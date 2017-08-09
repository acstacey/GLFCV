/**
 * This code has been adapted from:
 * https://github.com/atilimcetin/guided-filter
 * Copyright (c) 2014 Atilim Cetin
 * to use the OpenCV CUDA API.
 *
 * It implements the guided filter by Kaiming He (http://kaiminghe.com/eccv10/)
 *
 */

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

#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/opencv.hpp>

class GuidedFilterImpl;

class GuidedFilter {
 public:
  GuidedFilter(const cv::cuda::GpuMat &I, int r, double eps);
  ~GuidedFilter();

  cv::cuda::GpuMat filter(const cv::cuda::GpuMat &p, int depth = -1,
                          cv::cuda::Stream &stream = cv::cuda::Stream::Null()) const;

 private:
  GuidedFilterImpl *impl_;
};

cv::cuda::GpuMat guidedFilter(const cv::cuda::GpuMat &I,
                              const cv::cuda::GpuMat &p, int r,
                              double eps, int depth = -1,
                              cv::cuda::Stream &stream = cv::cuda::Stream::Null());

#endif
