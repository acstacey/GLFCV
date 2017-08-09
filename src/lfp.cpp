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

#include "lfp.h"

namespace lfp {

const size_t kSHA1Length = 45;
const size_t kBlankLength = 35;
const size_t kMinSectionLength = kSHA1Length + kBlankLength + 12 + 4;

const std::vector<uint8_t> kMagic{0x89, 0x4C, 0x46, 0x50, 0x0D, 0x0A,
                                  0x1A, 0x0A, 0x00, 0x00, 0x00, 0x01};
const std::string kTOCTypecode = "\x89LFM";

union SectionSize {
  uint8_t size_byte[4];
  uint32_t size_int;
};

Section::Section(const std::vector<uint8_t> &buffer, size_t &index) {
  // skip blank region between sections
  while (buffer[index] == 0) ++index;

  // copy 4-character type from 12-character string
  for (size_t i = 0; i < 12; ++i, ++index) {
    typecode.push_back((char)buffer[index]);
  }
  typecode = typecode.substr(0, 4);

  // copy section size
  SectionSize size;
  for (size_t i = 0; i < 4; ++i, ++index) {
    // convert big endian to little endian
    size.size_byte[3 - i] = buffer[index];
  }

  // copy SHA1 string
  for (size_t i = 0; i < kSHA1Length; ++i, ++index) {
    sha1.push_back((char)buffer[index]);
  }

  // skip blank length
  index += kBlankLength;

  // copy payload into data
  data = std::vector<uint8_t>(&buffer[index], &buffer[index + size.size_int]);
  index += size.size_int;
}

RawImage::RawImage(const Section &raw_image, const Section &metadata) {
  // parse metadata
  std::string metadata_str(metadata.data.begin(), metadata.data.end()),
      metadata_err;
  json11::Json metadata_json =
      json11::Json::parse_multi(metadata_str, metadata_err);


  json11::Json image = metadata_json[0]["image"];

  width = (size_t) image["width"].int_value();
  height = (size_t) image["height"].int_value();
  min_val = (uint32_t) image["pixelFormat"]["black"]["gr"].int_value();
  max_val = (uint32_t) image["pixelFormat"]["white"]["gr"].int_value();

  json11::Json devices = metadata_json[0]["devices"];

  focusStep = devices["lens"]["focusStep"].int_value();
  zoomStep = devices["lens"]["zoomStep"].int_value();

  bayer_pattern = devices["sensor"]["mosaic"]["tile"].string_value();

  rotation = devices["mla"]["rotation"].number_value();
  xMlaScale = devices["mla"]["scaleFactor"]["x"].number_value();
  yMlaScale = devices["mla"]["scaleFactor"]["y"].number_value();
  exitPupilOffsetZ = devices["lens"]["exitPupilOffset"]["z"].number_value();
  mlaSensorOffsetX = devices["mla"]["sensorOffset"]["x"].number_value();
  mlaSensorOffsetY = devices["mla"]["sensorOffset"]["y"].number_value();
  mlaSensorOffsetZ = devices["mla"]["sensorOffset"]["z"].number_value();
  pixelPitch = devices["sensor"]["pixelPitch"].number_value();
  lensPitch = devices["mla"]["lensPitch"].number_value();
  if (exitPupilOffsetZ != 0.0) {
    lensPitch *= (exitPupilOffsetZ + mlaSensorOffsetZ) / exitPupilOffsetZ;
  }
  if (pixelPitch == 0.0 || lensPitch == 0.0 || xMlaScale == 0.0 || yMlaScale == 0.0) {
    std::cerr << "Image metadata that should not have been zero was zero" << std::endl;
    std::abort();
  }

  // repack data to 16 bit
  size_t bit_depth =
      (size_t)metadata_json[0]["image"]["pixelPacking"]["bitsPerPixel"]
          .int_value();
  bool little_endian = !metadata_json[0]["image"]["pixelPacking"]["endianness"]
                            .string_value()
                            .compare("little");

  // TODO: Support more image and endian formats
  if (bit_depth != 10) {
    std::cerr << "Formats other than 10bit not supported" << std::endl;
    std::abort();
  }
  image_data = decode10bitData(raw_image.data, little_endian);
}

File::File(const std::string &filename) {
  // load file into memory
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> buffer((unsigned long)size);
  if (!file.read((char *)buffer.data(), size)) std::abort();

  // verify magic bytes
  bool valid = true;
  for (size_t i = 0; i < kMagic.size(); ++i) {
    valid &= (buffer[i] == kMagic[i]);
  }
  if (!valid) std::abort();

  // parse file sections
  size_t index = kMagic.size() + sizeof(uint32_t);
  Section *toc = nullptr;
  while (index + kMinSectionLength < size) {
    sections.push_back(Section(buffer, index));
    if (!sections.back().typecode.compare(kTOCTypecode)) {
      toc = &sections.back();
    }
  }
  if (!toc) std::abort();
  toc->name = "table";

  // parse table of contents json
  std::string toc_str(toc->data.begin(), toc->data.end()), toc_err;
  json11::Json toc_json = json11::Json::parse_multi(toc_str, toc_err);

  // find frames in metadata
  auto frame_names = toc_json[0]["frames"][0]["frame"].object_items();
  std::map<std::string, std::string> sha1_name_map;
  for (const auto &name : frame_names) {
    sha1_name_map[name.second.string_value()] = name.first;
  }

  // apply names to sections
  for (auto &section : sections) {
    const auto iter = sha1_name_map.find(section.sha1);
    if (iter != sha1_name_map.end()) {
      section.name = iter->second;
    }
  }

  // TODO: Apply other properties to sections

  // TODO: Parse jsons from json fields
}

RawImage File::GetRawImage() const {
  const Section *raw_data = nullptr, *metadata = nullptr;
  for (auto &section : sections) {
    if (!section.name.compare("metadataRef")) {
      metadata = &section;
    }
    if (!section.name.compare("imageRef")) {
      raw_data = &section;
    }
  }

  // return raw image
  if (raw_data == nullptr || metadata == nullptr) {
    std::cerr << "Couldn't load raw image" << std::endl;
    std::abort();
  }

  return RawImage(*raw_data, *metadata);
}

std::vector<uint16_t> decode10bitData(const std::vector<uint8_t> &data,
                                      bool little_endian) {
  if (!little_endian) {
    std::cerr << "big endian not supported" << std::endl;
    std::abort();
  }
  // NOTE: 10-bit format is packed as follows for every 5 bytes:
  // 0        1        2        3        4
  // MSB(A)   MSB(B)   MSB(C)   MSB(D)   LSB
  // AAAAAAAA BBBBBBBB CCCCCCCC DDDDDDDD DDCCBBAA

  size_t final_size = (data.size() * 8) / 10;
  std::vector<uint16_t> decodedData = std::vector<uint16_t>(final_size);
  for (size_t i = 0, j = 0; j < final_size; i += 5, j += 4) {
    // copy MSB from first four bytes
    for (size_t k = 0; k < 4; ++k) {
      decodedData[j + k] = data[i + k] << 2;
    }

    // copy LSB from last byte
    for (size_t k = 0; k < 4; ++k) {
      decodedData[j + k] += (data[i + 4] & (0x03 << 2 * k)) >> 2 * k;
    }
  }
  return decodedData;
}
}
