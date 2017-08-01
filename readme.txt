This code is licensed under GNU GPL V3, with a commercial licence available on request.
See LICENSE.txt for the full license text.

Build instructions:
	Install libraries as per required-library-installation.txt

	cd ./build
	cmake ..
	make

Running:

	Can be run on a folder with the image formats as in the benchmark datasets
	or on an LFR or LFP lytro file with the calibration archive

	Visualisations can be added by commenting in lines 74 and 75 of main.cpp

	LightFieldDepth <input>.LFR <white_image_folder> <output_folder>
	LightFieldDepth <light_field_folder> <output_dir>


	Example:
	cd ./build
	./LightFieldDepth ../data/cotton .

	OR for lytro images with decode:
	./LightFieldDepth ../data/IMG_0128.LFR ../data/caldata-B5155000720/ .
