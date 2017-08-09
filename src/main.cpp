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

#include <iostream>

#define BOOST_FILESYSTEM_NO_DEPRECATED
#include <boost/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "decoder.h"
#include "lf-depth-est.h"
#include "helper.h"

#define DISPLAY_RESULTS 1

void PrintUsage() {
  std::cout << "USAGE:    LightFieldDepth <input>.lfp <white_image_dir> <output_dir>"
            << std::endl;
  std::cout << "OR:    LightFieldDepth <light_field_folder> <output_dir>"
            << std::endl;
}

int main(int argc, char *argv[]) {
  // check correct number of arguments
  if (argc != 3 && argc != 4) {
    PrintUsage();
    return 1;
  }

  boost::filesystem::path input_path(argv[1]);
  std::string output_dir;

  std::vector<std::vector<cv::Mat>> lf;
  cv::Mat disp_ground_truth;
  cv::Mat disparity_map;

  struct image_meta metadata;
  metadata.isValid = false;

  if (boost::filesystem::is_directory(input_path)) {
    std::cout << "Input is a directory; searching for benchmark light field."
              << std::endl;

    output_dir = std::string(argv[2]);
    if (output_dir.back() != '/') {
      output_dir.push_back('/');
    }

    if (!decoder::ReadBenchmarkFolder(input_path, lf, disp_ground_truth, &metadata)) {
      std::cerr << "Failed to read light field directory" << std::endl;
      return 1;
    }
  } else {
    // parse input and output filenames
    std::string input_file(argv[1]);
    std::string caldata_dir(argv[2]);

    output_dir = std::string(argv[3]);

    std::string input_image_name;
    input_image_name = boost::filesystem::path(input_file).stem().string();

    if (caldata_dir.back() != '/') {
      caldata_dir.push_back('/');
    }
    if (output_dir.back() != '/') {
      output_dir.push_back('/');
    }

    // load and decode light fields
    double decodeStart = CPU_TIME();
    decoder::Decoder decoder(input_file, caldata_dir);
    decoder.Decode();
    std::cout << "Light field decode time: " << CPU_TIME() - decodeStart << " seconds" << std::endl;

    // display light field
    //decoder.DisplayLightFieldSlices("LFSlices", 50);
    //decoder.DisplayLenslet(kWindowName);

    // write to output file
    //decoder.WriteLensletImage(output_dir + input_image_name + "_lenslet.png", false);
    decoder.WriteLightFieldSlices(output_dir, input_image_name);

    lf = decoder.GetLightField();
  }

  if (metadata.isValid) {
    std::cout << "Input: disp_min = " << metadata.disparity_min << ", disp_max = " << metadata.disparity_max
              << std::endl;
  }

  // Hack to fix that the benchmark disparity is opposite of my shift values
  float old_min = metadata.disparity_min;
  float old_max = metadata.disparity_max;
  metadata.disparity_min = old_max * -1;//-3.2f;
  metadata.disparity_max = old_min * -1;//2.1f;

  double start_time = CPU_TIME();
  double disp_time;
  if (metadata.isValid) {
    BuildLFDisparityMap(lf, disparity_map, metadata.disparity_min, metadata.disparity_max);
  } else {
    BuildLFDisparityMap(lf, disparity_map);
  }
  disp_time = CPU_TIME() - start_time;
  std::cout << "Total disparity map creation time: " << disp_time << " seconds" << std::endl;

  boost::filesystem::path pfm_save(output_dir);
  pfm_save /= input_path.stem();
  pfm_save.replace_extension(".pfm");
  decoder::WriteCVMatToPFM(pfm_save, disparity_map);

  pfm_save.replace_extension(".txt");
  std::FILE *time_file = fopen(pfm_save.string().c_str(), "w");
  if (time_file != NULL) {
    fprintf(time_file, "%.5f\n", disp_time);
    fclose(time_file);
  }

  double disp_min, disp_max;
  cv::minMaxLoc(disparity_map, &disp_min, &disp_max);

  if (!disp_ground_truth.empty()) {
    double gt_min, gt_max;
    cv::Mat disparity_error = cv::abs(disparity_map - disp_ground_truth);
    cv::Mat error_pixels = disparity_error > 0.7;
    std::cout << "Number of pixels error: " << cv::countNonZero(error_pixels) << std::endl;

    cv::minMaxLoc(disp_ground_truth, &gt_min, &gt_max);

    if (disp_min < gt_min) gt_min = disp_min;
    if (disp_max > gt_max) gt_max = disp_max;

    disparity_map -= gt_min;
    disp_ground_truth -= gt_min;
    disparity_map.convertTo(disparity_map, disparity_map.type(),
                        1.0 / (gt_max - gt_min));
    disp_ground_truth.convertTo(disp_ground_truth, disp_ground_truth.type(),
                                 1.0 / (gt_max - gt_min));
#if DISPLAY_RESULTS
    decoder::DisplayCVMat("Disparity Map", disparity_map, 1);
    decoder::DisplayCVMat("Disparity Ground Truth", disp_ground_truth, 1);
    decoder::DisplayCVMat("Disparity Error", disparity_error, 1);
#endif
    disparity_map.convertTo(disparity_map, CV_16UC1,
                        (double) std::numeric_limits<uint16_t>::max());
    disp_ground_truth.convertTo(disp_ground_truth, CV_16UC1,
                                 (double) std::numeric_limits<uint16_t>::max());
    disparity_error.convertTo(disparity_error, CV_16UC1,
                              (double) std::numeric_limits<uint16_t>::max());
    cv::imwrite(output_dir + "disparity_map.png", disparity_map);
    cv::imwrite(output_dir + "disparity_map_gt.png", disp_ground_truth);
    cv::imwrite(output_dir + "disparity_error.png", disparity_error);
  } else {
    disparity_map -= disp_min;
    disparity_map.convertTo(disparity_map, disparity_map.type(),
                        1.0 / (disp_max - disp_min));
#if DISPLAY_RESULTS
    decoder::DisplayCVMat("Depth Map", disparity_map, 1);
#endif
    disparity_map.convertTo(disparity_map, CV_16UC1, (double) std::numeric_limits<uint16_t>::max());
    cv::imwrite(output_dir + "disparity_map.png", disparity_map);
  }

  // return success
  return 0;
}