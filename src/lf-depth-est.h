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


#ifndef LIGHTFIELDOPENCV_LF_DEPTH_EST_H
#define LIGHTFIELDOPENCV_LF_DEPTH_EST_H

#include <opencv2/core/core.hpp>

// Alpha values recommended by Tao-DFC-2013
//#define ALPHA_MIN 0.2f
//#define ALPHA_MAX 2.0f

#if defined(ALPHA_MIN) && defined(ALPHA_MAX)
#define SHIFT_MIN (1.0 - (1.0 / ALPHA_MIN))
#define SHIFT_MAX (1.0 - (1.0 / ALPHA_MAX))
#else
#define SHIFT_MIN -3.0f
#define SHIFT_MAX 2.0f
#endif

#define DEPTH_RESOLUTION 128

#define GUIDED_FILTER_NEIGHBOURHOOD 4 //4
#define GUIDED_FILTER_SMOOTHING 1e-3  //1e-3
#define SOBEL_KERNEL_SIZE 1

#define TAD_C_TAO 0.1f
#define TAD_G_TAO 0.1f
#define TAD_CG_ALPHA 0.25f

#define NUM_CUDA_STREAMS 2

void BuildLFDisparityMap(const std::vector<std::vector<cv::Mat> > &lf, cv::Mat &disparity_map,
                         float disp_min = SHIFT_MIN, float disp_max = SHIFT_MAX);

#endif //LIGHTFIELDOPENCV_LF_DEPTH_EST_H
