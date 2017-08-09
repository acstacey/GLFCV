# GLFCV - Light field disparity estimation using a guided filter cost volume

Guided filter Light Field Cost Volume is a CUDA implementation of a disparity estimation algorithm for 4D light fields.
It uses the [guided filter](http://kaiminghe.com/eccv10/) on cost volume slices computed using the [TAD C+G metric](http://www.sciencedirect.com/science/article/pii/S1077314213000143) on 4D shears of the light field.  Estimated disparity is calculated by an argmin on the filtered cost volume over a range of disparity values.

### License
This code is licensed under GNU GPL V3, with a commercial licence available on request.
See LICENSE file for the full license text.

Copyright (C) 2017 Adam Stacey



## Build instructions
Install libraries as per last section of README.

```
cd ./build
cmake ..
make
```

## Usage

GLFCV can be run on a folder with the image formats as in the benchmark datasets
or on an LFR or LFP lytro file with the calibration archive.

Visualisations can be added by commenting in lines 74 and 75 of main.cpp

```
GLFCV <input>.LFR <white_image_folder> <output_folder>
GLFCV <light_field_folder> <output_dir>
```


### Examples
#### HCI Benchmark Scene
```
cd ./build
./GLFCV ../data/cotton .
```
The following images are the results of GLFCV on the Cotton benchmark scene.
Left to right: central sub-aperture image, GLFCV disparity estimate, ground truth, GLFCV error vs ground truth.
<div align="center">
<kbd>
<img src="/../screenshots/screenshots/cotton-image.png?raw=true" width="200">
<img src="/../screenshots/screenshots/cotton-GLFCV-res.png?raw=true" width="200">
<img src="/../screenshots/screenshots/cotton-gt.png?raw=true" width="200">
<img src="/../screenshots/screenshots/cotton-error.png?raw=true" width="200">
</kbd>
</div>


#### Lytro Images With Decoding
```
cd ./build
./GLFCV ../data/IMG_0128.LFR ../data/caldata-B5155000720/ .
```


## Library installation:

### Ubuntu 16.04:

#### CUDA
Download cuda local .deb from NVIDIA
```
sudo dpkg -i cuda*.deb
sudo apt-get update
sudo apt-get install cuda

export PATH=/usr/local/cuda-8.0.61/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-8.0.61/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
#### OpenCV
```
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/3.2.0
mkdir build
```
Use cmake-gui or cmake to generate makefiles including with cuda, jpeg and qt5 and generate build files in the build directory
```
cd build
make -j7
sudo make install
```

#### Boost
```
sudo apt install libboost-all-dev
```

### macOS
#### CUDA
Download and run installer https://developer.nvidia.com/cuda-downloads
```
export PATH=/Developer/NVIDIA/CUDA-8.0/bin${PATH:+:${PATH}}
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
```

#### OpenCV
```
brew install opencv3 --with-contrib --with-cuda --with-qt5 --with-libtiff --with-eigen
```

#### Boost
Install boost 1.58 or higher
```
brew install boost
brew switch boost 1.58.0  # Some systems need 1.58.0 specifically
```
