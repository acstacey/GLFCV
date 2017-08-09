#GLFCV - Light field disparity estimation using a guided filter cost volume

This code is licensed under GNU GPL V3, with a commercial licence available on request.
See LICENSE for the full license text.

##Build instructions
	Install libraries as per last section of README.

    ```
	cd ./build
	cmake ..
	make
	```

##Running

	GLFCV can be run on a folder with the image formats as in the benchmark datasets
	or on an LFR or LFP lytro file with the calibration archive

	Visualisations can be added by commenting in lines 74 and 75 of main.cpp

	```
	GLFCV <input>.LFR <white_image_folder> <output_folder>
	GLFCV <light_field_folder> <output_dir>
	```


	###Example
	```
	cd ./build
	./GLFCV ../data/cotton .
	```
	OR for lytro images with decode:
	```
	./GLFCV ../data/IMG_0128.LFR ../data/caldata-B5155000720/ .
	```


##Library installation:

###Ubuntu 16.04:

    ####CUDA
	Download cuda local .deb from NVIDIA
	```
	sudo dpkg -i cuda*.deb
	sudo apt-get update
	sudo apt-get install cuda

	export PATH=/usr/local/cuda-8.0.61/bin${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda-8.0.61/lib64\ ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	```

	####OpenCV
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

	####Boost
	```
	sudo apt install libboost-all-dev
	```

###macOS
	####CUDA
	Download and run installer https://developer.nvidia.com/cuda-downloads
	```
	export PATH=/Developer/NVIDIA/CUDA-8.0/bin${PATH:+:${PATH}}
	export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-8.0/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}
	```

	####OpenCV
	```
	brew install opencv3 --with-contrib --with-cuda --with-qt5 --with-libtiff --with-eigen
	```

	####Boost
	Install boost 1.58 or higher
	```
	brew install boost
	brew switch boost 1.58.0  # Some systems need 1.58.0 specifically
	```