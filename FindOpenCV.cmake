###########################################################
#                  Find OpenCV Library
# See http://sourceforge.net/projects/opencvlibrary/
#----------------------------------------------------------
#
## 1: Setup:
# The following variables are optionally searched for defaults
#  OpenCV_DIR:            Base directory of OpenCv tree to use.
#
## 2: Variable
# The following are set after configuration is done: 
#  
#  OpenCV_FOUND
#  OpenCV_LIBS
#  OpenCV_INCLUDE_DIR
#  OpenCV_VERSION (OpenCV_VERSION_MAJOR, OpenCV_VERSION_MINOR, OpenCV_VERSION_PATCH)
#
#
# Deprecated variable are used to maintain backward compatibility with
# the script of Jan Woetzel (2006/09): www.mip.informatik.uni-kiel.de/~jw
#  OpenCV_INCLUDE_DIRS
#  OpenCV_LIBRARIES
#  OpenCV_LINK_DIRECTORIES
# 
## 3: Version
#
# 2016/08/04 Daniel Wilde, Assume OpenCVConfig.cmake exists and does most of the work
# 2010/04/07 Benoit Rat, Correct a bug when OpenCVConfig.cmake is not found.
# 2010/03/24 Benoit Rat, Add compatibility for when OpenCVConfig.cmake is not found.
# 2010/03/22 Benoit Rat, Creation of the script.
#
#
# tested with:
# - OpenCV 2.1:  MinGW, MSVC2008
# - OpenCV 2.0:  MinGW, MSVC2008, GCC4
#
#
## 4: Licence:
#
# LGPL 2.1 : GNU Lesser General Public License Usage
# Alternatively, this file may be used under the terms of the GNU Lesser

# General Public License version 2.1 as published by the Free Software
# Foundation and appearing in the file LICENSE.LGPL included in the
# packaging of this file.  Please review the following information to
# ensure the GNU Lesser General Public License version 2.1 requirements
# will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.
# 
#----------------------------------------------------------

# Optionally specify OpenCV_DIR
find_path(OpenCV_DIR "share/OpenCV/OpenCVConfig.cmake" 
          DOC "Root directory of OpenCV")

if(NOT EXISTS "${OpenCV_DIR}")
  #Find OpenCV using FindPkgConfig instead
  if(NOT WIN32)
    include(FindPkgConfig)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(OPENCV_PKGCONF opencv)
      set(OpenCV_DIR ${OPENCV_PKGCONF_PREFIX} CACHE PATH "" FORCE)
      if(EXISTS "${OpenCV_DIR}")
	      set(OpenCV_configScript_DIR "${OpenCV_DIR}/share/OpenCV")
	      if(EXISTS "${OpenCV_configScript_DIR}")
	        set(OpenCV_configScript "${OpenCV_configScript_DIR}/OpenCVConfig.cmake")
	      endif()
      endif()
    endif()
  endif()
else()
  # Assume OpenCV_DIR is correct
  set(OpenCV_configScript "${OpenCV_DIR}/share/OpenCV/OpenCVConfig.cmake")
endif()

##====================================================
## Find OpenCV libraries
##----------------------------------------------------
if(EXISTS "${OpenCV_configScript}")
  #This will define OpenCV_LIBS
  include("${OpenCV_configScript}")
endif()

find_path(OpenCV_INCLUDE_DIRS "cv.h" PATHS "${OpenCV_DIR}" 
          PATH_SUFFIXES "include" "include/opencv" DOC "Include directory") 
find_path(OpenCV_INCLUDE_DIR "cv.h" PATHS "${OpenCV_DIR}" 
          PATH_SUFFIXES "include" "include/opencv" DOC "Include directory")
mark_as_advanced(FORCE OpenCV_INCLUDE_DIRS OpenCV_INCLUDE_DIR)
set(OpenCV_LIBRARIES "${OpenCV_LIBS}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCV
  FOUND_VAR OpenCV_FOUND
  REQUIRED_VARS OpenCV_INCLUDE_DIR OpenCV_LIBS
  VERSION_VAR OpenCV_VERSION)
  