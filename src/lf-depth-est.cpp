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


#include "lf-depth-est.h"
#include "decoder.h"
#include "guidedfilter.h"
#include "helper.h"

#include <cuda_profiler_api.h>

#include "cuda-kernels/lf-cuda-kernels.h"

#define PERFORM_GUIDED_FILTER_ITER 1
#define PERFORM_GUIDED_FILTER_FINAL 1

#define MONO_COLOUR 0

/**
 * GPU disparity calculation
 * @param lf
 * @param disparity_map
 * @param disp_min
 * @param disp_max
 */
void BuildLFDisparityMap(const std::vector<std::vector<cv::Mat>> &lf, cv::Mat &disparity_map,
                         float disp_min, float disp_max) {

  setCudaParamsForImage(lf[0][0], cuda_threads_per_block);
  cudaProfilerStart();

  // Maintain cuda stream vectors in both OpenCV and native types
  std::vector<cv::cuda::Stream> streams(NUM_CUDA_STREAMS);
  std::vector<cudaStream_t> c_streams;
  for (auto s : streams) {
    c_streams.push_back(cv::cuda::StreamAccessor::getStream(s));
  }

  double start_time = CPU_TIME();
  cv::cuda::setDevice(0);

  // Transfer light field to the GPU
  std::vector<std::vector<cv::cuda::GpuMat>> lf_gpu(lf.size());
  for (int i = 0; i < lf.size(); ++i) {
    lf_gpu[i] = std::vector<cv::cuda::GpuMat>(lf[0].size());
  }
  for (int i = 0; i < lf.size(); ++i) {
    for (int j = 0; j < lf[0].size(); ++j) {
      lf_gpu[i][j].upload(lf[i][j], streams[j % NUM_CUDA_STREAMS]);
#if MONO_COLOUR
      cv::cuda::cvtColor(lf_gpu[i][j], lf_gpu[i][j], cv::COLOR_BGR2GRAY, 1, streams[j % NUM_CUDA_STREAMS]);
#endif
    }
  }

  double shift_step = (disp_max - disp_min) / (DEPTH_RESOLUTION - 0.5);

  // Angular coordinates
  size_t u_size = lf.size();
  size_t v_size = lf[0].size();
  float min_u_shift, min_v_shift;
  min_u_shift = u_size / 2;
  min_v_shift = v_size / 2;
  min_u_shift = u_size % 2 == 1 ? -min_u_shift : -min_u_shift + 0.5f;
  min_v_shift = v_size % 2 == 1 ? -min_v_shift : -min_v_shift + 0.5f;

  const double num_sub_images = u_size * v_size;

  streams[0].waitForCompletion();
  cv::cuda::GpuMat reference_image = lf_gpu[u_size / 2][v_size / 2];

  std::vector<cv::cuda::GpuMat> x_maps_gpu, y_maps_gpu, defoc_parts, dx, dy,
      defoc_gradients, tad_c_images, tad_cg_means, tad_cg_costs;
  std::vector<std::vector<cv::cuda::GpuMat>> split_channel_sets;

  // preallocate intermediate variables
  x_maps_gpu.resize(NUM_CUDA_STREAMS, cv::cuda::GpuMat(lf_gpu[0][0].rows,
                                                       lf_gpu[0][0].cols, CV_32FC1));
  y_maps_gpu.resize(NUM_CUDA_STREAMS, cv::cuda::GpuMat(lf_gpu[0][0].rows,
                                                       lf_gpu[0][0].cols, CV_32FC1));
  split_channel_sets.resize(NUM_CUDA_STREAMS);
  dx.resize(NUM_CUDA_STREAMS);
  dy.resize(NUM_CUDA_STREAMS);
  defoc_parts.resize(NUM_CUDA_STREAMS);
  defoc_gradients.resize(NUM_CUDA_STREAMS);
  tad_c_images.resize(NUM_CUDA_STREAMS, cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols, lf_gpu[0][0].type()));
  tad_cg_means.resize(NUM_CUDA_STREAMS, cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols, lf_gpu[0][0].type()));

  tad_cg_costs.resize(DEPTH_RESOLUTION, cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols, lf_gpu[0][0].type()));

  cv::cuda::GpuMat min_max_mask = cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols, CV_32FC1);

  cv::Ptr<cv::cuda::Filter> filter_dx = cv::cuda::createSobelFilter(
      reference_image.type(), reference_image.type(), 1, 0, SOBEL_KERNEL_SIZE);
  cv::Ptr<cv::cuda::Filter> filter_dy = cv::cuda::createSobelFilter(
      reference_image.type(), reference_image.type(), 0, 1, SOBEL_KERNEL_SIZE);
  filter_dx->apply(reference_image, dx[0]);
  filter_dy->apply(reference_image, dy[0]);
  cv::cuda::GpuMat reference_gradient;
  cv::cuda::addWeighted(dx[0], 0.5, dy[0], 0.5, 0, reference_gradient);

  cv::cuda::GpuMat tad_cg_curr_min = cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols, CV_32FC1,
                                                      cv::Scalar(std::numeric_limits<float>::max()));
  cv::cuda::GpuMat tad_cg_disp = cv::cuda::GpuMat(lf_gpu[0][0].rows, lf_gpu[0][0].cols,
                                                  CV_32FC1, cv::Scalar(0.0));

#if PERFORM_GUIDED_FILTER_ITER
  GuidedFilter iter_filter(reference_image, GUIDED_FILTER_NEIGHBOURHOOD, GUIDED_FILTER_SMOOTHING);
#endif
  std::cout << "Filter creation, reference grad and lf to GPU transfer: " << CPU_TIME() - start_time << " seconds"
            << std::endl;

  double shifting_tadcg_time = 0;
  double mean_cost_time = 0;
  double argmin_time = 0;
  std::cout << "'=' printed every 10 shifts.  " << DEPTH_RESOLUTION << " total shifts" << std::endl;

  // Compute responses for each alpha value and produce disparity map for both
  size_t depth_index = 0;
  std::vector<float> shifts;
  for (float shift = disp_min; shift <= disp_max; shift += shift_step) {
    shifts.push_back(shift);
    int stream_idx = (int) (depth_index % NUM_CUDA_STREAMS);
    if (depth_index % 10 == 0) {
      std::cout << "=";
      std::cout.flush();
    }

    tad_cg_means[stream_idx].setTo(cv::Scalar(0.0), streams[stream_idx]);

    start_time = CPU_TIME();
    // Compute shift for each sub aperture image and add to defoc image
    for (int u = 0; u < u_size; ++u) {
      for (int v = 0; v < v_size; ++v) {
        ShiftMapCalc(x_maps_gpu[stream_idx],
                     y_maps_gpu[stream_idx],
                     (min_v_shift + v) * shift,
                     (min_u_shift + u) * shift,
                     c_streams[stream_idx]);
        cv::cuda::remap(lf_gpu[u][v], defoc_parts[stream_idx], x_maps_gpu[stream_idx], y_maps_gpu[stream_idx],
                        cv::INTER_LINEAR, cv::BORDER_REPLICATE, cv::Scalar(), streams[stream_idx]);

        filter_dx->apply(defoc_parts[stream_idx], dx[stream_idx], streams[stream_idx]);
        filter_dy->apply(defoc_parts[stream_idx], dy[stream_idx], streams[stream_idx]);

#if MONO_COLOUR
        ContribToTadCGMeanBW(tad_cg_means[stream_idx], reference_image,
                                 reference_gradient, defoc_parts[stream_idx],
                                 dx[stream_idx], dy[stream_idx], c_streams[stream_idx]);
#else
        ContribToTadCGMeanColour(tad_cg_means[stream_idx], reference_image,
                                 reference_gradient, defoc_parts[stream_idx],
                                 dx[stream_idx], dy[stream_idx], c_streams[stream_idx]);
#endif
      }
    }
    shifting_tadcg_time += CPU_TIME() - start_time;

    start_time = CPU_TIME();

    cv::cuda::divide(tad_cg_means[stream_idx],
                     cv::Scalar(num_sub_images),
                     tad_cg_means[stream_idx],
                     1,
                     -1,
                     streams[stream_idx]);

#if MONO_COLOUR
    tad_cg_means[stream_idx].copyTo(tad_cg_costs[depth_index], streams[stream_idx]);
#else
    // Used to average each pixel because
    // cvtcolor uses 0.299*R + 0.587*G + 0.114*B
    split_channel_sets[stream_idx].clear();
    cv::cuda::split(tad_cg_means[stream_idx], split_channel_sets[stream_idx], streams[stream_idx]);
    cv::cuda::add(split_channel_sets[stream_idx][0],
                  split_channel_sets[stream_idx][1],
                  tad_cg_costs[depth_index],
                  cv::noArray(),
                  -1,
                  streams[stream_idx]);
    cv::cuda::add(tad_cg_costs[depth_index],
                  split_channel_sets[stream_idx][2],
                  tad_cg_costs[depth_index],
                  cv::noArray(),
                  -1,
                  streams[stream_idx]);
    cv::cuda::divide(tad_cg_costs[depth_index], cv::Scalar(3.0), tad_cg_costs[depth_index], 1, -1, streams[stream_idx]);
    tad_cg_costs[depth_index] =
        tad_cg_costs[depth_index].clone(); // bad disparity maps from overwritten memory without this
#endif

#if PERFORM_GUIDED_FILTER_ITER
    tad_cg_costs[depth_index] = iter_filter.filter(tad_cg_costs[depth_index], -1, streams[stream_idx]);
#endif

    mean_cost_time += CPU_TIME() - start_time;

    ++depth_index;
  }

  for (auto s : streams) s.waitForCompletion();

  start_time = CPU_TIME();
  // Build disparity map from argmin
  for (int d = 0; d < DEPTH_RESOLUTION; ++d) {
    cv::cuda::compare(tad_cg_costs[d], tad_cg_curr_min, min_max_mask, cv::CMP_LT,
                      streams[0]);
    tad_cg_costs[d].copyTo(tad_cg_curr_min, min_max_mask, streams[0]);
    tad_cg_disp.setTo(cv::Scalar(-shifts[d]), min_max_mask, streams[0]);
  }
  streams[0].waitForCompletion();
  argmin_time += CPU_TIME() - start_time;

#if PERFORM_GUIDED_FILTER_FINAL
  guidedFilter(tad_cg_disp, tad_cg_disp, GUIDED_FILTER_NEIGHBOURHOOD, GUIDED_FILTER_SMOOTHING);
#endif
  tad_cg_disp.download(disparity_map);

  cudaProfilerStop();
  std::cout << std::endl;
  std::cout << "Shifting and tad_cg calc: " << shifting_tadcg_time << " seconds" << std::endl;
  std::cout << "Mean and cost calc: " << mean_cost_time << " seconds" << std::endl;
  std::cout << "Argmin calc: " << argmin_time << " seconds" << std::endl;

}

