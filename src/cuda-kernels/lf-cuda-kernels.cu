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

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "lf-cuda-kernels.h"
#include "../lf-depth-est.h"

dim3 cuda_threads_per_block(16, 16);
dim3 cuda_num_blocks(32, 32); //temporary hard code for 512 * 512 images

/**
 * Arrange CUDA blocks to accomodate image.  Can be optimised further.
 * @param image_cols
 * @param image_rows
 * @param threads
 */
void setCudaParamsForImage(size_t image_cols, size_t image_rows, dim3 threads) {
  cuda_threads_per_block.x = threads.x;
  cuda_threads_per_block.y = threads.y;
  cuda_threads_per_block.z = threads.z;
  cuda_num_blocks = dim3(
      static_cast<int>(std::ceil(image_cols /
          static_cast<double>(cuda_threads_per_block.x))),
      static_cast<int>(std::ceil(image_rows /
          static_cast<double>(cuda_threads_per_block.y))));
}

void setCudaParamsForImage(cv::Mat image, dim3 threads) {
  setCudaParamsForImage(image.cols, image.rows, threads);
}

void setCudaParamsForImage(cv::cuda::GpuMat image, dim3 threads) {
  setCudaParamsForImage(image.cols, image.rows, threads);
}

/**
 * Populate a gpuMat for a u,v shift when computing the 4D shear of the light field
 * @param x_map_gpu
 * @param y_map_gpu
 * @param v_shift
 * @param u_shift
 */
__global__ void ShiftMapCalcKernel(cv::cuda::PtrStepSzf x_map_gpu,
                                   cv::cuda::PtrStepSzf y_map_gpu,
                                   float v_shift,
                                   float u_shift) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < x_map_gpu.cols && y < x_map_gpu.rows && y >= 0 && x >= 0) {
    x_map_gpu(y, x) = x + v_shift;
    y_map_gpu(y, x) = y + u_shift;
  }
}

void ShiftMapCalc(cv::cuda::PtrStepSzf x_map_gpu,
                  cv::cuda::PtrStepSzf y_map_gpu,
                  float v_shift,
                  float u_shift,
                  cudaStream_t stream) {
  ShiftMapCalcKernel << < cuda_num_blocks, cuda_threads_per_block, 0, stream >> > (x_map_gpu, y_map_gpu,
      v_shift, u_shift);
}


/**
 * Calculate the TAD C+G contribution for a sheared light field (colour)
 * @param t_cg_m
 * @param ref
 * @param ref_grad
 * @param defoc
 * @param dx
 * @param dy
 */
__global__ void ContribToTadCGMeanColourKernel(cv::cuda::PtrStepSzf t_cg_m,
                                               cv::cuda::PtrStepSzf ref,
                                               cv::cuda::PtrStepSzf ref_grad,
                                               cv::cuda::PtrStepSzf defoc,
                                               cv::cuda::PtrStepSzf dx,
                                               cv::cuda::PtrStepSzf dy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float temp_grad0, temp_grad1, temp_grad2;
  if (x < t_cg_m.cols && y < t_cg_m.rows && y >= 0 && x >= 0) {

    temp_grad0 = fminf(fabsf(ref_grad(y, (3 * x)) - (0.5f * dx(y, (3 * x)) + 0.5f * dy(y, (3 * x)))), TAD_G_TAO);
    temp_grad1 =
        fminf(fabsf(ref_grad(y, (3 * x) + 1) - (0.5f * dx(y, (3 * x) + 1) + 0.5f * dy(y, (3 * x) + 1))), TAD_G_TAO);
    temp_grad2 =
        fminf(fabsf(ref_grad(y, (3 * x) + 2) - (0.5f * dx(y, (3 * x) + 2) + 0.5f * dy(y, (3 * x) + 2))), TAD_G_TAO);
    t_cg_m(y, (3 * x)) += (TAD_CG_ALPHA * fminf(fabsf(ref(y, (3 * x)) - defoc(y, (3 * x))), TAD_C_TAO)) +
        ((1 - TAD_CG_ALPHA) * temp_grad0);
    t_cg_m(y, (3 * x) + 1) += (TAD_CG_ALPHA * fminf(fabsf(ref(y, (3 * x) + 1) - defoc(y, (3 * x) + 1)), TAD_C_TAO)) +
        ((1 - TAD_CG_ALPHA) * temp_grad1);
    t_cg_m(y, (3 * x) + 2) += (TAD_CG_ALPHA * fminf(fabsf(ref(y, (3 * x) + 2) - defoc(y, (3 * x) + 2)), TAD_C_TAO)) +
        ((1 - TAD_CG_ALPHA) * temp_grad2);
  }
}

void ContribToTadCGMeanColour(cv::cuda::PtrStepSzf t_cg_m,
                              cv::cuda::PtrStepSzf ref,
                              cv::cuda::PtrStepSzf ref_grad,
                              cv::cuda::PtrStepSzf defoc,
                              cv::cuda::PtrStepSzf dx,
                              cv::cuda::PtrStepSzf dy, cudaStream_t stream) {
  ContribToTadCGMeanColourKernel << < cuda_num_blocks, cuda_threads_per_block, 0, stream >> > (t_cg_m, ref,
      ref_grad, defoc, dx, dy);
}

/**
 * Calculate the TAD C+G contribution for a sheared light field (grayscale)
 * @param t_cg_m
 * @param ref
 * @param ref_grad
 * @param defoc
 * @param dx
 * @param dy
 */
__global__ void ContribToTadCGMeanBWKernel(cv::cuda::PtrStepSzf t_cg_m,
                                           cv::cuda::PtrStepSzf ref,
                                           cv::cuda::PtrStepSzf ref_grad,
                                           cv::cuda::PtrStepSzf defoc,
                                           cv::cuda::PtrStepSzf dx,
                                           cv::cuda::PtrStepSzf dy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  float temp_grad;
  if (x < t_cg_m.cols && y < t_cg_m.rows && y >= 0 && x >= 0) {

    temp_grad = fminf(fabsf(ref_grad(y, (3 * x)) - (0.5f * dx(y, x) + 0.5f * dy(y, x))), TAD_G_TAO);
    t_cg_m(y, x) += (TAD_CG_ALPHA * fminf(fabsf(ref(y, x) - defoc(y, x)), TAD_C_TAO)) +
        ((1 - TAD_CG_ALPHA) * temp_grad);
  }
}

void ContribToTadCGMeanBW(cv::cuda::PtrStepSzf t_cg_m,
                          cv::cuda::PtrStepSzf ref,
                          cv::cuda::PtrStepSzf ref_grad,
                          cv::cuda::PtrStepSzf defoc,
                          cv::cuda::PtrStepSzf dx,
                          cv::cuda::PtrStepSzf dy, cudaStream_t stream) {
  ContribToTadCGMeanBWKernel << < cuda_num_blocks, cuda_threads_per_block, 0, stream >> > (t_cg_m, ref, ref_grad,
      defoc, dx, dy);
}

__global__ void AddColourKernel(cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> b) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < a.cols && y < a.rows && y >= 0 && x >= 0) {
    a(y, (3 * x) + 0) += b(y, (3 * x) + 0);
    a(y, (3 * x) + 1) += b(y, (3 * x) + 1);
    a(y, (3 * x) + 2) += b(y, (3 * x) + 2);
  }
}

void AddColour(cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> b) {
  AddColourKernel << < cuda_num_blocks, cuda_threads_per_block >> > (a, b);
}

__global__ void AbsdiffColourKernel(cv::cuda::PtrStepSz<float> a,
                                    cv::cuda::PtrStepSz<float> b,
                                    cv::cuda::PtrStepSz<float> output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < a.cols && y < a.rows && y >= 0 && x >= 0) {
    output(y, (3 * x) + 0) = fabsf(a(y, (3 * x) + 0) - b(y, (3 * x) + 0));
    output(y, (3 * x) + 1) = fabsf(a(y, (3 * x) + 1) - b(y, (3 * x) + 1));
    output(y, (3 * x) + 2) = fabsf(a(y, (3 * x) + 2) - b(y, (3 * x) + 2));
  }
}

void AbsdiffColour(cv::cuda::PtrStepSz<float> a, cv::cuda::PtrStepSz<float> b,
                   cv::cuda::PtrStepSzf output, cudaStream_t stream) {
  AbsdiffColourKernel << < cuda_num_blocks, cuda_threads_per_block, 0, stream >> > (a, b, output);
}
