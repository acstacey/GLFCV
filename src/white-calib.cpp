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

#include "lfp.h"
#include "white-calib.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace whiteCalib {

std::pair<size_t, size_t> FindWhiteImageIndex(
    const json11::Json &mla_calib_json, int desired_zoom_step,
    int desired_foc_step);

LensletGridModel::LensletGridModel() {
  exitPupilDistanceMilli = 0.0;
  xMlaSensorOffsetMicrons = 0.0;
  yMlaSensorOffsetMicrons = 0.0;
  xMlaScale = 0.0;
  yMlaScale = 0.0;
  xDiskOriginPixels = 0.0;
  yDiskOriginPixels = 0.0;
  xDiskStepPixelsX = 0.0;
  xDiskStepPixelsY = 0.0;
  yDiskStepPixelsX = 0.0;
  yDiskStepPixelsY = 0.0;
  rotation = 0.0;
  horizOrient = true;
}

WhiteImage::WhiteImage(const std::string &calDataDir, int desired_zoom_step,
                       int desired_foc_step) {
  // Load the Json description of properties for white images
  std::ifstream mla_calib_in(calDataDir + DEFAULT_MLA_CALIB_FILE);
  std::string mla_calib_options((std::istreambuf_iterator<char>(mla_calib_in)),
                                (std::istreambuf_iterator<char>()));
  mla_calib_in.close();

  std::string json_err;
  json11::Json mla_calib_json =
      json11::Json::parse(mla_calib_options, json_err);

  // Match the white image focus and zoom
  std::cout << "Finding white image to match - zoom: " << desired_zoom_step << " focus: " << desired_foc_step << std::endl;
  std::pair<size_t, size_t> zoom_foc_indices =
      FindWhiteImageIndex(mla_calib_json, desired_zoom_step, desired_foc_step);
  image_index = 0;
  for (size_t i = 0; i < zoom_foc_indices.first; ++i) {
    image_index += mla_calib_json["numFocusPositions"][i].int_value();
  }
  image_index += zoom_foc_indices.second;

  zoomStep = mla_calib_json["zoomPosition"][zoom_foc_indices.first].int_value();
  focusStep = mla_calib_json["focusPosition"][zoom_foc_indices.first]
                            [zoom_foc_indices.second].int_value();

  std::cout << "Using white image " << image_index << " - zoom: " << zoomStep << " focus: " << focusStep << std::endl;
  mla_grid_model = LensletGridModel();

  mla_grid_model.exitPupilDistanceMilli =
      mla_calib_json["exitPupilDistanceMilli"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.rotation =
      mla_calib_json["rotation"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.xMlaSensorOffsetMicrons =
      mla_calib_json["xMlaSensorOffsetMicrons"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.yMlaSensorOffsetMicrons =
      mla_calib_json["yMlaSensorOffsetMicrons"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.xMlaScale =
      mla_calib_json["xMlaScale"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.yMlaScale =
      mla_calib_json["yMlaScale"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.xDiskOriginPixels =
      mla_calib_json["xDiskOriginPixels"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.yDiskOriginPixels =
      mla_calib_json["yDiskOriginPixels"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.xDiskStepPixelsX =
      mla_calib_json["xDiskStepPixelsX"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.xDiskStepPixelsY =
      mla_calib_json["xDiskStepPixelsY"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.yDiskStepPixelsX =
      mla_calib_json["yDiskStepPixelsX"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();
  mla_grid_model.yDiskStepPixelsY =
      mla_calib_json["yDiskStepPixelsY"][zoom_foc_indices.first]
                    [zoom_foc_indices.second].number_value();

  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << image_index;
  std::string fileNumber = ss.str();

  // Read in chosen white image metadata
  std::ifstream white_meta_in(calDataDir + DEFAULT_FILE_PREFIX + fileNumber +
                              DEFAULT_META_FILETYPE);
  std::string white_meta((std::istreambuf_iterator<char>(white_meta_in)),
                         (std::istreambuf_iterator<char>()));
  white_meta_in.close();
  json11::Json white_meta_json = json11::Json::parse(white_meta, json_err);
  white_meta_json = white_meta_json["master"]["picture"]["frameArray"][0]
                                   ["frame"]["metadata"];

  width = white_meta_json["image"]["width"].int_value();
  height = white_meta_json["image"]["height"].int_value();
  /*
  bayer_pattern =
      white_meta_json["image"]["rawDetails"]["mosaic"]["tile"].string_value();
  */
  min_val = white_meta_json["image"]["rawDetails"]["pixelFormat"]["black"]["r"]
                .int_value();
  max_val = white_meta_json["image"]["rawDetails"]["pixelFormat"]["white"]["r"]
                .int_value();
  exposureDuration =
      white_meta_json["devices"]["shutter"]["frameExposureDuration"]
          .number_value();

  // Look for Light field toolbox processing on the white images
  used_lftoolbox_calib = false;
  std::ifstream lf_toolbox_grid_in(calDataDir + DEFAULT_FILE_PREFIX +
      fileNumber + DEFAULT_TOOLBOX_FILETYPE);
  if (lf_toolbox_grid_in.good()) {
    used_lftoolbox_calib = true;
    std::cout << "Found white image calibration from LFToolbox" << std::endl;
    std::string toolbox_content(
        (std::istreambuf_iterator<char>(lf_toolbox_grid_in)),
        (std::istreambuf_iterator<char>()));
    json11::Json toolbox_json = json11::Json::parse(toolbox_content, json_err);
    toolbox_json = toolbox_json["LensletGridModel"];
    mla_grid_model.xDiskStepPixelsX = toolbox_json["HSpacing"].number_value();
    mla_grid_model.yDiskStepPixelsY = toolbox_json["VSpacing"].number_value();
    mla_grid_model.xDiskOriginPixels = toolbox_json["HOffset"].number_value();
    mla_grid_model.yDiskOriginPixels = toolbox_json["VOffset"].number_value();
    mla_grid_model.rotation = toolbox_json["Rot"].number_value();
    if (toolbox_json["FirstPosShiftRow"].int_value() == 1) {
      mla_grid_model.xDiskOriginPixels += mla_grid_model.xDiskStepPixelsX / 2.0;
    }
  }
  lf_toolbox_grid_in.close();

  // Read raw white image
  std::ifstream file(
      calDataDir + DEFAULT_FILE_PREFIX + fileNumber + DEFAULT_IMAGE_FILETYPE,
      std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer((unsigned long)size);
  if (!file.read((char *)buffer.data(), size)) std::abort();

  if (white_meta_json["image"]["rawDetails"]["pixelPacking"]["bitsPerPixel"]
          .int_value() != 10) {
    std::cerr << "Formats other than 10bit not supported" << std::endl;
    std::abort();
  }
  bool little_endian =
      !white_meta_json["image"]["rawDetails"]["pixelPacking"]["endianness"]
           .string_value()
           .compare("little");
  image_data = lfp::decode10bitData(buffer, little_endian);
}

/**
 * Finds the zoom and focus step indices of the white image whose zoom and focus steps
 * are closest to the desired ones.
 * Matching is done first by zoom step, then by focus step.  In the case of two equally close zoom steps,
 * the one with the closest matching focus step is used.  In the case of two equally matching focus steps,
 * the lower one is used.
 * It is assumed that the zoom and focus steps are sorted in the calibration json
 * @param mla_calib_json
 * @param desired_zoom_step
 * @param desired_foc_step
 * @return (zoom index, focus index) pair
 */
std::pair<size_t, size_t> FindWhiteImageIndex(
    const json11::Json &mla_calib_json, int desired_zoom_step,
    int desired_foc_step) {
  size_t i, zoom_index, focus_index, curr_diff,
      min_diff = std::numeric_limits<size_t>::max();
  int secondary_zoom_index = -1;
  bool use_secondary_zoom = false;

  i = 0;
  for (auto zoom : mla_calib_json["zoomPosition"].array_items()) {
    if (desired_zoom_step < zoom.int_value()) {
      curr_diff = zoom.int_value() - desired_zoom_step;
    } else {
      curr_diff = desired_zoom_step - zoom.int_value();
    }
    if (curr_diff < min_diff) {
      zoom_index = i;
      secondary_zoom_index = -1;
      min_diff = curr_diff;
    } else if (curr_diff == min_diff) {
      secondary_zoom_index = i;
    }
    ++i;
  }

  i = 0;
  min_diff = std::numeric_limits<size_t>::max();
  for (auto focus : mla_calib_json["focusPosition"][zoom_index].array_items()) {
    if (desired_foc_step < focus.int_value()) {
      curr_diff = focus.int_value() - desired_foc_step;
    } else {
      curr_diff = desired_foc_step - focus.int_value();
    }
    if (curr_diff < min_diff) {
      focus_index = i;
      min_diff = curr_diff;
    }
    ++i;
  }

  i = 0;
  if (secondary_zoom_index >= 0) {
    for (auto focus :
         mla_calib_json["focusPosition"][secondary_zoom_index].array_items()) {
      if (desired_foc_step < focus.int_value()) {
        curr_diff = focus.int_value() - desired_foc_step;
      } else {
        curr_diff = desired_foc_step - focus.int_value();
      }
      if (curr_diff < min_diff) {
        focus_index = i;
        min_diff = curr_diff;
        use_secondary_zoom = true;
      }
      ++i;
    }
  }

  if (use_secondary_zoom) zoom_index = secondary_zoom_index;

  return std::make_pair(zoom_index, focus_index);
}
}
