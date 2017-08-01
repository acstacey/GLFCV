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


#ifndef LIGHTFIELDOPENCV_LF_CUDA_KERNELS_H
#define LIGHTFIELDOPENCV_LF_CUDA_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

extern dim3 cuda_threads_per_block;
extern dim3 cuda_num_blocks;

void setCudaParamsForImage(cv::cuda::GpuMat image, dim3 threads);
void setCudaParamsForImage(cv::Mat image, dim3 threads);
void setCudaParamsForImage(cv::cuda::GpuMat image, dim3 threads);

void ShiftMapCalc(const cv::cuda::PtrStepSzf x_map_gpu,
                  const cv::cuda::PtrStepSzf y_map_gpu,
                  float v_shift,
                  float u_shift,
                  cudaStream_t stream);

void AddColour(cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> b);

void ContribToTadCGMeanColour(cv::cuda::PtrStepSzf t_cg_m,
                              cv::cuda::PtrStepSzf ref,
                              cv::cuda::PtrStepSzf ref_grad,
                              cv::cuda::PtrStepSzf defoc,
                              cv::cuda::PtrStepSzf dx,
                              cv::cuda::PtrStepSzf dy, cudaStream_t stream);

void ContribToTadCGMeanBW(cv::cuda::PtrStepSzf t_cg_m,
                          cv::cuda::PtrStepSzf ref,
                          cv::cuda::PtrStepSzf ref_grad,
                          cv::cuda::PtrStepSzf defoc,
                          cv::cuda::PtrStepSzf dx,
                          cv::cuda::PtrStepSzf dy, cudaStream_t stream);

void AbsdiffColour(cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> b,
                   cv::cuda::PtrStepSzf output, cudaStream_t stream);

#endif //LIGHTFIELDOPENCV_LF_CUDA_KERNELS_H
