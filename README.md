opticflow
=========

This repository contains code for estimating optic flow by using a Nvidia GPU. It is a modification of the example provided by CUDA library in which is implemented the model of Horn & Schunck. Hence, this software contains source code provided by NVIDIA Corporation.

The code has been tested in OpenSUSE 13.1 and Ubuntu 14.04 with Nvidia GPU's with compute capabilities 2.1. However, I am confident this piece of code can be executed in other linux distributions without problems. In order to compile the code provided, you need to be sure that OpenCV 2.4 and CUDA 6.0 or later are installed in your system.

COMPILATION:

After git clone, you need to modify the Makefile provided in order to match your linux system:

You can try with an earlier CUDA library by modifying:
CUDEFS += -DCUDA_TOOLKIT_6
CUDA_INSTALL_PATH = /usr/local/cuda-6.0

The pkg-config package must be installed in order to provide information about the OpenCV library. In the case of missing pkg-config, you can modify the variables CXXFLAGS and LIBS, setting manually the OpenCV cflags and libs.
CXXFLAGS :=	${CXXFLAGS} ${DEFS} $(shell pkg-config --cflags opencv)
LIBS = -pthread $(shell pkg-config --libs opencv) 

The Makefile provided can manage authomatically all the development files (.cpp, .c, .cu, .cuh) present in the SRC folder. In case you want to use a complementary source folder, you can add it by modifying:
DIRECTORIES = src
