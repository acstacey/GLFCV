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

#include <fstream>
#include <iostream>

#include <eigen3/Eigen/Eigen>

#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "decoder.h"

//#define DEBUG

// Used for metadata lines in pfm files
#define LINE_BUFFER_LENGTH 80

// Number of pixels away from lenslets to sample sub-aperture images
#define LENSLET_SLICE_MAX_OFFSET 4

// Adjustments used when no Matlab Light Field Toolbox calibration is available
#define NO_TOOLBOX_X_OFFSET 1.5
#define NO_TOOLBOX_Y_OFFSET 1.0
#define NO_TOOLBOX_X_SCALE_POWER 2.0
#define NO_TOOLBOX_Y_SCALE_POWER 2.0

namespace decoder {

bool ReadPFMToCV(boost::filesystem::path &pfm_loc, cv::Mat &image,
                 bool allow_blank_lines);

Decoder::Decoder(const std::string &filename, const std::string &calDataDir)
    :  // Load LFR and extract raw image
    raw_data_(lfp::File(filename).GetRawImage()),
    // Find and load appropriate white image
    white_data_(calDataDir, raw_data_.zoomStep, raw_data_.focusStep) {
  // convert to cv::mat
  raw_image_ = cv::Mat(raw_data_.height, raw_data_.width, CV_16U,
                       raw_data_.image_data.data())
      .clone();
  white_image_ = cv::Mat(white_data_.height, white_data_.width, CV_16U,
                         white_data_.image_data.data())
      .clone();
}

/**
 * Use raw lenslet image and white image to sample a 2D array of sub-aperture images
 */
void Decoder::Decode() {
  // rescale according to min and max values
  float offset = (float) white_data_.min_val;
  float scale =
      (float) std::numeric_limits<uint16_t>::max()
          / (float) white_data_.max_val;
  cv::Mat white_image_float;
  white_image_.convertTo(white_image_float, CV_32F, scale, offset);

  offset = (float) raw_data_.min_val;
  scale =
      (float) std::numeric_limits<uint16_t>::max() / (float) raw_data_.max_val;
  cv::Mat raw_image_float;
  raw_image_.convertTo(raw_image_float, CV_32F, scale, offset);

  // Use white image to scale intensities before demosaic
  std::cout << "Devignette with white image" << std::endl;
  raw_image_float = raw_image_float / white_image_float;
  raw_image_float.convertTo(raw_image_, CV_16U,
                            std::numeric_limits<uint16_t>::max(), 0);

  // perform bayer demosaic
  std::cout << "Performing bayer demosaic" << std::endl;
  if (raw_data_.bayer_pattern.compare("r,gr:gb,b")) {  // GRGB bayer pattern
    std::cerr << "Bayer pattern not supported" << std::endl;
    std::abort();
  }

  decoded_lenslet_image_ = cv::Mat(raw_image_.rows, raw_image_.cols, CV_16UC3);
  cv::cvtColor(raw_image_, decoded_lenslet_image_, CV_BayerGB2RGB);

  // build interpolation grids
  double output_scale = 1.0; //2.0 / 1.7312;
  double max_grid_step = std::max(white_data_.mla_grid_model.xDiskStepPixelsX,
                                  white_data_.mla_grid_model.yDiskStepPixelsY) /
      output_scale;
  size_t grid_width = std::floor((double) raw_data_.width / max_grid_step);
  size_t grid_height = std::floor((double) raw_data_.height / max_grid_step);
  std::vector<cv::Mat> grid_interp_locations, grid_interp_weights;
  for (size_t i = 0; i < 3; ++i) {
    grid_interp_locations.push_back(cv::Mat(grid_height, grid_width, CV_32FC2));
    grid_interp_weights.push_back(cv::Mat(grid_height, grid_width, CV_32FC3));
  }

  cv::Point2d grid_origin(white_data_.mla_grid_model.xDiskOriginPixels,
                          white_data_.mla_grid_model.yDiskOriginPixels);

  cv::Point2d grid_step(white_data_.mla_grid_model.xMlaScale *
                            white_data_.mla_grid_model.xDiskStepPixelsX,
                        white_data_.mla_grid_model.yMlaScale *
                            white_data_.mla_grid_model.yDiskStepPixelsY);

  // Without the more precise calibration from the MATLAB light field toolbox
  // some experimental factors improve accuracy of lenslet centres
  if (!white_data_.used_lftoolbox_calib) {
    std::cout << "Applying corrections without LFToolbox data" << std::endl;
    grid_origin.x += NO_TOOLBOX_X_OFFSET;
    grid_origin.y += NO_TOOLBOX_Y_OFFSET;
    grid_step.x = std::pow(
        white_data_.mla_grid_model.xMlaScale, NO_TOOLBOX_X_SCALE_POWER) *
        white_data_.mla_grid_model.xDiskStepPixelsX;
    grid_step.y = std::pow(
        white_data_.mla_grid_model.yMlaScale, NO_TOOLBOX_Y_SCALE_POWER) *
        white_data_.mla_grid_model.yDiskStepPixelsY;
  }

  // invert rotation angle to derotate grid
  double rotation[4] = {std::cos(-white_data_.mla_grid_model.rotation),
                        -std::sin(-white_data_.mla_grid_model.rotation),
                        std::sin(-white_data_.mla_grid_model.rotation),
                        std::cos(-white_data_.mla_grid_model.rotation)};
  cv::Mat grid_rotation = cv::Mat(2, 2, CV_64F, &rotation).clone();
  double inv_rotation[4] = {std::cos(white_data_.mla_grid_model.rotation),
                            -std::sin(white_data_.mla_grid_model.rotation),
                            std::sin(white_data_.mla_grid_model.rotation),
                            std::cos(white_data_.mla_grid_model.rotation)};
  cv::Mat grid_inv_rotation = cv::Mat(2, 2, CV_64F, &inv_rotation).clone();

#ifdef DEBUG
  // debug file output
  std::cout << "Initialising debug files" << std::endl;
  std::ofstream grid_coordinates_file, triangle_coordinates_file,
      triangle_weights_file;
  grid_coordinates_file.open("../../output-data/grid_coordinates.csv",
                             std::ofstream::out | std::ofstream::trunc);
  triangle_coordinates_file.open("../../output-data/triangle_coordinates.csv",
                                 std::ofstream::out | std::ofstream::trunc);
  triangle_weights_file.open("../../output-data/triangle_weights.csv",
                             std::ofstream::out | std::ofstream::trunc);
#endif
  // iterate over grid cells and build triangle interpolants
  std::cout << "Building interpolation triangles with lenslet centres"
            << std::endl;
  for (size_t row = 0; row < grid_height; ++row) {
    for (size_t col = 0; col < grid_width; ++col) {
      // get uncorrected subpixel location in input
      cv::Point2d raw_subpix((col + 0.5) * max_grid_step,
                             (row + 0.5) * max_grid_step);
#ifdef DEBUG
      grid_coordinates_file << raw_subpix.x << ", " << raw_subpix.y
                            << std::endl;
#endif

      // subtract origin and derotate
      cv::Point2d transformed_subpix = cv::Point2d(
          grid_rotation.at<double>(0, 0) * (raw_subpix.x - grid_origin.x) +
              grid_rotation.at<double>(0, 1) * (raw_subpix.y - grid_origin.y),
          grid_rotation.at<double>(1, 0) * (raw_subpix.x - grid_origin.x) +
              grid_rotation.at<double>(1, 1) * (raw_subpix.y - grid_origin.y));

      // compute position within triangle grid
      cv::Point2d transformed_idx =
          cv::Point2d(transformed_subpix.x / grid_step.x,
                      transformed_subpix.y / grid_step.y);
      cv::Point2i triangle_idx = cv::Point2i(std::floor(transformed_idx.x),
                                             std::floor(transformed_idx.y));
      cv::Point2d remainder_idx =
          cv::Point2d(transformed_idx.x - triangle_idx.x,
                      transformed_idx.y - triangle_idx.y);

      // triangle grid positions:
      //       0     1
      // 0:    A-----B
      //      / \   / \
      //     /   \ /   \
      // 1: C-----D-----E
      //     \   / \   /
      //      \ /   \ /
      // 2:    F-----G

      // select triangle vertices based on grid position
      std::vector<cv::Point2d> triangle_vertices;
      switch (std::abs(triangle_idx.y) % 2) {
        case 0: {  // even index - ACD, ADB, BDE
          cv::Point2d A(grid_step.x * triangle_idx.x,
                        grid_step.y * triangle_idx.y);
          cv::Point2d B(grid_step.x * (triangle_idx.x + 1),
                        grid_step.y * triangle_idx.y);
          cv::Point2d C(grid_step.x * (triangle_idx.x - 0.5),
                        grid_step.y * (triangle_idx.y + 1));
          cv::Point2d D(grid_step.x * (triangle_idx.x + 0.5),
                        grid_step.y * (triangle_idx.y + 1));
          cv::Point2d E(grid_step.x * (triangle_idx.x + 1.5),
                        grid_step.y * (triangle_idx.y + 1));

          if (remainder_idx.y > 2 * remainder_idx.x) {
            // if point above AD, choose ACD
            triangle_vertices.push_back(A);
            triangle_vertices.push_back(C);
            triangle_vertices.push_back(D);
          } else if (remainder_idx.y > 2 * (1 - remainder_idx.x)) {
            // if point above BD, choose BDE
            triangle_vertices.push_back(B);
            triangle_vertices.push_back(D);
            triangle_vertices.push_back(E);
          } else {
            // else choose ADB
            triangle_vertices.push_back(A);
            triangle_vertices.push_back(D);
            triangle_vertices.push_back(B);
          }

        }
          break;
        case 1: {  // odd index - CFD, DFG, DGE
          cv::Point2d C(grid_step.x * (triangle_idx.x - 0.5),
                        grid_step.y * triangle_idx.y);
          cv::Point2d D(grid_step.x * (triangle_idx.x + 0.5),
                        grid_step.y * triangle_idx.y);
          cv::Point2d E(grid_step.x * (triangle_idx.x + 1.5),
                        grid_step.y * triangle_idx.y);
          cv::Point2d F(grid_step.x * triangle_idx.x,
                        grid_step.y * (triangle_idx.y + 1));
          cv::Point2d G(grid_step.x * (triangle_idx.x + 1),
                        grid_step.y * (triangle_idx.y + 1));

          if (remainder_idx.y < 1 - 2 * remainder_idx.x) {
            // if point below DF, choose CFD
            triangle_vertices.push_back(C);
            triangle_vertices.push_back(F);
            triangle_vertices.push_back(D);
          } else if (remainder_idx.y < 2 * remainder_idx.x - 1) {
            // if point below DG, choose DGE
            triangle_vertices.push_back(D);
            triangle_vertices.push_back(G);
            triangle_vertices.push_back(E);
          } else {
            // else choose DFG
            triangle_vertices.push_back(D);
            triangle_vertices.push_back(F);
            triangle_vertices.push_back(G);
          }

        }
          break;
        default:std::abort();
          break;
      }

      // compute weights from triangle vertices
      cv::Point3d triangle_weights;
      triangle_weights.x = TriangleArea(
          triangle_vertices[1], triangle_vertices[2], transformed_subpix);
      triangle_weights.y = TriangleArea(
          triangle_vertices[0], triangle_vertices[2], transformed_subpix);
      triangle_weights.z = TriangleArea(
          triangle_vertices[0], triangle_vertices[1], transformed_subpix);

      // normalise weights
      double weight_sum =
          triangle_weights.x + triangle_weights.y + triangle_weights.z;
      triangle_weights.x /= weight_sum;
      triangle_weights.y /= weight_sum;
      triangle_weights.z /= weight_sum;

#ifdef DEBUG
      triangle_weights_file << triangle_weights.x << ", " << triangle_weights.y
                            << ", " << triangle_weights.z << std::endl;
#endif

      // transform triangle corners back to original grid
      std::vector<cv::Point2d> raw_triangle_vertices;
      for (size_t i = 0; i < 3; ++i) {
        cv::Point2d transformed_vertex(
            grid_inv_rotation.at<double>(0, 0) * triangle_vertices[i].x +
                grid_inv_rotation.at<double>(0, 1) * triangle_vertices[i].y +
                grid_origin.x,
            grid_inv_rotation.at<double>(1, 0) * triangle_vertices[i].x +
                grid_inv_rotation.at<double>(1, 1) * triangle_vertices[i].y +
                grid_origin.y);
        raw_triangle_vertices.push_back(transformed_vertex);
#ifdef DEBUG
        triangle_coordinates_file << transformed_vertex.x << ", "
                                  << transformed_vertex.y << ", ";
#endif
      }
#ifdef DEBUG
      triangle_coordinates_file << std::endl;
#endif


      // write values to grid interpolation images
      for (size_t i = 0; i < 3; ++i) {
        grid_interp_locations[i].at<cv::Vec2f>(cv::Point(col, row))[0] =
            (float) (raw_triangle_vertices[i].x);
        grid_interp_locations[i].at<cv::Vec2f>(cv::Point(col, row))[1] =
            (float) (raw_triangle_vertices[i].y);
      }
      grid_interp_weights[0].at<cv::Vec3f>(cv::Point(col, row)) =
          cv::Vec3f(triangle_weights.x, triangle_weights.x, triangle_weights.x);
      grid_interp_weights[1].at<cv::Vec3f>(cv::Point(col, row)) =
          cv::Vec3f(triangle_weights.y, triangle_weights.y, triangle_weights.y);
      grid_interp_weights[2].at<cv::Vec3f>(cv::Point(col, row)) =
          cv::Vec3f(triangle_weights.z, triangle_weights.z, triangle_weights.z);
    }
  }
#ifdef DEBUG
  grid_coordinates_file.close();
  triangle_coordinates_file.close();
  triangle_weights_file.close();
#endif

  light_field_ = std::vector<std::vector<cv::Mat>>();
  std::cout << "Interpolating lenslet slice images" << std::endl;
  for (int row_offset = -LENSLET_SLICE_MAX_OFFSET;
       row_offset < LENSLET_SLICE_MAX_OFFSET + 1;
       ++row_offset) {
    light_field_.push_back(std::vector<cv::Mat>());
    for (int col_offset = -LENSLET_SLICE_MAX_OFFSET;
         col_offset < LENSLET_SLICE_MAX_OFFSET + 1;
         ++col_offset) {
      cv::Mat out_image_sum = cv::Mat(grid_height, grid_width, CV_32FC3);
      out_image_sum = cv::Scalar::all(0.0);
      for (size_t i = 0; i < 3; ++i) {
        cv::Mat out_image = cv::Mat(grid_height, grid_width, CV_16UC3);

        // Add (x, y) scalar to grid_interp_locations[i] to offset from lenslet
        // centres
        cv::Mat grid_interp_locations_offset = grid_interp_locations[i].clone();
        cv::add(grid_interp_locations_offset,
                cv::Scalar(row_offset, col_offset),
                grid_interp_locations_offset);
        cv::remap(decoded_lenslet_image_,
                  out_image,
                  grid_interp_locations_offset,
                  cv::Mat(),
                  cv::INTER_LINEAR);
        out_image.convertTo(out_image, CV_32FC3,
                            1.0 / (double) std::numeric_limits<uint16_t>::max());
        out_image_sum += out_image.mul(grid_interp_weights[i]);
      }
      light_field_.back().push_back(out_image_sum.clone());
    }
  }
  std::cout << "Decoding complete" << std::endl;
}

/**
 * Cycle display through all sub-aperture images.
 * Finishing on the central view.
 * @param window_name - name of UI window
 * @param refresh_speed - milliseconds spent showing each image
 */
void Decoder::DisplayLightFieldSlices(const std::string &window_name,
                                      int refresh_speed) const {
  if (light_field_.empty()) {
    std::cerr << "Can't display light field slices - not decoded yet"
              << std::endl;
    return;
  }
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::Mat display_image = cv::Mat(light_field_[0][0].rows,
                                  light_field_[0][0].cols,
                                  light_field_[0][0].type());
  for (auto const &row : light_field_) {
    for (auto const &sub_image : row) {
      cv::cvtColor(sub_image, display_image, CV_RGB2BGR);
      cv::imshow(window_name, display_image);
      cv::waitKey(refresh_speed);

    }
  }
  cv::cvtColor(light_field_[light_field_.size() / 2][light_field_.size() / 2],
               display_image,
               CV_RGB2BGR);
  cv::imshow(window_name, display_image);
  cv::waitKey(0);
  cv::destroyWindow(window_name);
}

void Decoder::DisplayLenslet(const std::string &window_name) const {
  if (!decoded_lenslet_image_.empty()) {
    DisplayCVMat(window_name, decoded_lenslet_image_, 0.2);
  } else {
    DisplayCVMat(window_name, raw_image_, 0.2);
  }
}

/**
 * Save lenslet image to a file
 * @param filename
 * @param raw
 */
void Decoder::WriteLensletImage(const std::string &filename,
                                const bool raw) const {
  // convert to BGR format for writing if necessary
  cv::Mat output_image;
  if (raw) {
    output_image = raw_image_;
  } else {
    output_image = cv::Mat(decoded_lenslet_image_.rows,
                           decoded_lenslet_image_.cols,
                           CV_16UC3);
    cv::cvtColor(decoded_lenslet_image_, output_image, CV_RGB2BGR);
  }

  // write output image to file
  cv::imwrite(filename, output_image);
  std::cout << "Wrote " << (raw ? "raw" : "decoded") << " lenslet image to "
            << filename << std::endl;
}

/**
 * Saves all sub-aperture images with their sub-pixel coordinates appended
 * @param out_dir
 * @param image_name
 */
void Decoder::WriteLightFieldSlices(const std::string &out_dir,
                                    const std::string &image_name) const {
  if (light_field_.empty()) {
    std::cerr << "Can't output light field slices - not decoded yet"
              << std::endl;
    return;
  }
  std::string mkdir_command = "mkdir -p " + out_dir + image_name;
  system(mkdir_command.c_str());
  cv::Mat output_image =
      cv::Mat(light_field_[0][0].rows, light_field_[0][0].cols, CV_16UC3);
  for (size_t row = 0; row < light_field_.size(); ++row) {
    for (size_t col = 0; col < light_field_[row].size(); ++col) {
      light_field_[row][col].convertTo(output_image,
                                       CV_16UC3,
                                       std::numeric_limits<uint16_t>::max());
      cv::cvtColor(output_image, output_image, CV_RGB2BGR);
      cv::imwrite(out_dir + image_name + "/" + image_name + "_" +
          std::to_string(row + 1) + "_" +
          std::to_string(col + 1) + ".png", output_image);
    }
  }
}

const std::vector<std::vector<cv::Mat>> &Decoder::GetLightField() {
  return light_field_;
}

/**
 * Load the light field, ground truth, and metadata (disparity range)
 * from a folder matching the structure used by the HCI light field benchmark
 * @param input_dir
 * @param lf
 * @param depth_truth
 * @param meta
 * @return true if an HCI benchmark folder was correctly read
 */
bool ReadBenchmarkFolder(boost::filesystem::path &input_dir,
                         std::vector<std::vector<cv::Mat>> &lf,
                         cv::Mat &depth_truth, struct image_meta *meta) {
  const std::string depth_name = "gt_disp_lowres.pfm";
  const std::string meta_name = "parameters.cfg";
  boost::filesystem::path depth_location(input_dir / depth_name);
  boost::filesystem::path meta_location(input_dir / meta_name);
  ReadPFMToCV(depth_location, depth_truth, true);

  if (boost::filesystem::exists(meta_location)) {
    std::ifstream meta_file(meta_location.string().c_str());
    if (meta_file.is_open()) {
      std::string line;
      std::string disp_min_name("disp_min");
      std::string disp_max_name("disp_max");
      while (std::getline(meta_file, line)) {
        if (line.compare(0, disp_min_name.length(), disp_min_name, 0, disp_min_name.length()) == 0) {
          line = line.substr(line.find_first_of("-0123456789."));
          std::stringstream info(line);
          info >> meta->disparity_min;
        } else if (line.compare(0, disp_max_name.length(), disp_max_name, 0, disp_max_name.length()) == 0) {
          line = line.substr(line.find_first_of("-0123456789."));
          std::stringstream info(line);
          info >> meta->disparity_max;
        }
      }
      meta->isValid = true;
      meta_file.close();
    } else {
      std::cout << "Benchmark parameters file did not open" << std::endl;
    }
  } else {
    std::cout << "Benchmark parameters not found" << std::endl;
  }

  std::vector<std::string> sub_image_paths;
  const boost::regex filter(".*input_Cam.*\\.png");

  boost::filesystem::directory_iterator end_itr;
  for (boost::filesystem::directory_iterator i(input_dir); i != end_itr; ++i) {
    // Skip if not a file
    if (boost::filesystem::is_regular_file(i->status())) {
      boost::smatch what;

      if (boost::regex_search(i->path().string(), what, filter)) {
        sub_image_paths.push_back(i->path().string());
      }
    }
  }

  // should be a square number >2
  if (sub_image_paths.size() < 4) {
    return false;
  }

  size_t dimension = (size_t) std::sqrt(sub_image_paths.size());
  lf.clear();

  // make sure sub images are loaded in order of their index
  std::sort(sub_image_paths.begin(), sub_image_paths.end());

  double factor = 1.0; // used to scale images loaded with different types
  cv::Scalar blah;

  // load images
  cv::Mat sub_image;
  for (size_t u = 0; u < dimension; ++u) {
    lf.push_back(std::vector<cv::Mat>());
    for (size_t v = 0; v < dimension; ++v) {
      sub_image = cv::imread(sub_image_paths[u * dimension + v]);

      if (sub_image.empty()) return false;

      // Make floating point with range from 0-1
      switch (sub_image.type()) {
        case CV_8UC1:
        case CV_8UC2:
        case CV_8UC3:
        case CV_8UC4:factor = 1.0 / (double) std::numeric_limits<uint8_t>::max();
          break;
        case CV_16UC1:
        case CV_16UC2:
        case CV_16UC3:
        case CV_16UC4:factor = 1.0 / (double) std::numeric_limits<uint16_t>::max();
          break;
        case CV_32FC1:
        case CV_32FC2:
        case CV_32FC3:
        case CV_32FC4:
        case CV_64FC1:
        case CV_64FC2:
        case CV_64FC3:
        case CV_64FC4:factor = 1.0;
          break;
        default:
          std::cerr << "Unknown image type loaded: " << sub_image.type() <<
                    std::endl;
          std::abort();
      }

      sub_image.convertTo(sub_image, CV_32FC3, factor);
      lf.back().push_back(sub_image.clone());
    }
  }
  return true;
}

/**
 * Read pfm into cv::Mat.  Not tested for multi-channel pfm
 * @param pfm_loc
 * @param image
 * @param allow_blank_lines
 * @return
 */
bool ReadPFMToCV(boost::filesystem::path &pfm_loc, cv::Mat &image,
                 bool allow_blank_lines) {
  if (!boost::filesystem::exists(pfm_loc)) return false;

  unsigned int num_chans = 0; // 1 for grayscale, 3 for RGB
  unsigned int width = 0;
  unsigned int height = 0;
  float scale = 0.0;
  bool is_big_endian = false;

  char line1[LINE_BUFFER_LENGTH + 1];
  char line2[LINE_BUFFER_LENGTH + 1];
  char line3[LINE_BUFFER_LENGTH + 1];
  line1[0] = '\0';
  line2[0] = '\0';
  line3[0] = '\0';

  std::FILE *pfm_file = fopen(pfm_loc.string().c_str(), "r");
  if (pfm_file == NULL) return false;

  if (allow_blank_lines) {
    char COMMENT_CHAR = '#';
    std::fgets(line1, LINE_BUFFER_LENGTH, pfm_file);
    while (std::strlen(line1) == 0 || line1[0] == COMMENT_CHAR) {
      std::fgets(line1, LINE_BUFFER_LENGTH, pfm_file);
    }
    std::fgets(line2, LINE_BUFFER_LENGTH, pfm_file);
    while (std::strlen(line2) == 0 || line2[0] == COMMENT_CHAR) {
      std::fgets(line2, LINE_BUFFER_LENGTH, pfm_file);
    }
    std::fgets(line3, LINE_BUFFER_LENGTH, pfm_file);
    while (std::strlen(line3) == 0 || line3[0] == COMMENT_CHAR) {
      std::fgets(line3, LINE_BUFFER_LENGTH, pfm_file);
    }
  } else {
    std::fgets(line1, LINE_BUFFER_LENGTH, pfm_file);
    std::fgets(line2, LINE_BUFFER_LENGTH, pfm_file);
    std::fgets(line3, LINE_BUFFER_LENGTH, pfm_file);
  }

  if (line1[0] == 'P' && line1[1] == 'F') {
    num_chans = 3;
  } else if (line1[0] == 'P' && line1[1] == 'f') {
    num_chans = 1;
  } else {
    std::cerr << "PFM header line 1 was invalid. Line1: " << line1 << std::endl;
    std::fclose(pfm_file);
    return false;
  }

  if (std::sscanf(line2, "%u %u", &width, &height) != 2) {
    std::cerr << "PFM header line 2 was invalid. Line2: " << line2 << std::endl;
    std::fclose(pfm_file);
    return false;
  }

  if (std::sscanf(line3, "%f", &scale) != 1) {
    std::cerr << "PFM header line 3 was invalid. Line3: " << line3 << std::endl;
    std::fclose(pfm_file);
    return false;
  }

  float scale_factor = std::abs(scale);
  if (scale < 0.0) {
    is_big_endian = false;
  } else {
    is_big_endian = true;
  }

  size_t tot_elems = width * height * num_chans;

  cv::Mat tmp;
  if (num_chans == 3) {
    tmp = cv::Mat(height, width, CV_32FC3);
  } else {
    tmp = cv::Mat(height, width, CV_32FC1);
  }

  size_t elems_read = std::fread(tmp.data, 4, tot_elems, pfm_file);
  if (elems_read != tot_elems) {
    std::cerr << "PFM data didn't match header." << std::endl;
    std::fclose(pfm_file);
    return false;
  }
  std::fclose(pfm_file);
  size_t offset = 0;
  uchar temp_byte;

  // Correct for endianness mismatch if present
  if (is_big_endian != IsMachineBigEndian()) {
    for (size_t elem = 0; elem < tot_elems; ++elem) {
      offset = elem * 4;
      temp_byte = tmp.data[offset];
      tmp.data[offset] = tmp.data[offset + 3];
      tmp.data[offset + 3] = temp_byte;
      temp_byte = tmp.data[offset + 1];
      tmp.data[offset + 1] = tmp.data[offset + 2];
      tmp.data[offset + 2] = temp_byte;
    }
  }

  image = tmp.clone();

  // NOTE: might not work for 3 channel data
  // Because of openCV ordering it needs to be rotated 180 and mirrored
  cv::transpose(image, image);
  cv::flip(image, image, 0);
  cv::transpose(image, image);
  cv::flip(image, image, 0);
  cv::flip(image, image, 1);

  return true;
}

/**
 * Write cv::Mat to pfm.  Not tested for multi-channel images
 * @param save_loc
 * @param image
 * @return
 */
bool WriteCVMatToPFM(boost::filesystem::path &save_loc, cv::Mat &image) {

  cv::Mat output;
  char line_buffer[15];
  std::FILE *pfm = fopen(save_loc.string().c_str(), "w");
  if (pfm == NULL) return false;

  if (image.depth() == 3) {
    fputs("PF\n", pfm);
  } else {
    fputs("Pf\n", pfm);
  }

  snprintf(line_buffer, 15, "%d %d\n", image.cols, image.rows);
  fputs(line_buffer, pfm);

  if (IsMachineBigEndian()) {
    fputs("1.0\n", pfm);
  } else {
    fputs("-1.0\n", pfm);
  }

  output = image.clone();

  // NOTE: might not work for 3 channel data
  // Because of openCV ordering it needs to be rotated 180 and mirrored
  cv::transpose(output, output);
  cv::flip(output, output, 0);
  cv::transpose(output, output);
  cv::flip(output, output, 0);
  cv::flip(output, output, 1);

  if (!output.isContinuous()) {
    output = output.clone();
  }

  fwrite(output.data, sizeof(float), output.cols * output.rows, pfm);

  fclose(pfm);
  return true;

}

/**
 * Display a cv::Mat as an image, scaled by param downscale
 * @param window_name
 * @param image
 * @param downscale
 */
void DisplayCVMat(const std::string &window_name, const cv::Mat &image,
                  double downscale) {
  // convert to BGR format for display
  cv::Mat display_image = cv::Mat(image.rows, image.cols, image.type());
  if (image.channels() == 3) {
    cv::cvtColor(image, display_image, CV_RGB2BGR);
  } else {
    display_image = image;
  }

  cv::resize(display_image, display_image, cv::Size(), downscale, downscale,
             CV_INTER_AREA);

  // display in a window until a key is pressed
  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, display_image);
  cv::waitKey(0);
  cv::destroyWindow(window_name);
}

}
