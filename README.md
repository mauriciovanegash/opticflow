opticflow
=========

This repository contains code for estimating optic flow by using a Nvidia GPU. It is a modification of the example provided by CUDA library in which is implemented the model of Horn & Schunck. Hence, this software contains source code provided by NVIDIA Corporation.

The code has been tested in OpenSUSE 13.1 and Ubuntu 14.04 with Nvidia GPU's with compute capabilities 2.1. However, I am confident this piece of code can be executed in other linux distributions. In order to compile the code provided, you need to be sure that OpenCV 2.4.0 and CUDA 6.0 or later are installed in your system.

HOW IT WORKS:

The code provided will use any webcam available in the system. The class PDESolver provides two methods for allocating memory, one for allocating unified memory and the other for allocating device memory. The class deals with the deallocation of any pointer allocated by the methods provided, and therefore, it is not necessary or even convenient do it manually. The example opticflow.cpp will estimate the optic flow by using the code HSOpticalFlow (into CUDA Samples) with a slight modification in order to render it computationally efficient. In HSOpticalFlow, it has been implemented the iterative solver Jacobi method which has a fixed time step size. It is known that a fixed time step size is rather inefficient due to a severe stability restriction. Hence, I modified the simple Jacobi iteration and rendered it a Fast Jacobi iteration in oder to allow cycles of varying step sizes as proposed by Grewenig et al. I used the public code FED/FJ Library available at http://www.mia.uni-saarland.de/Research/SC_FED.shtml. For more details regards Fast Jacobi method please refer to [Grewenig et al. 2013, Grewenig et al. 2010] in BIBLIOGRAPHY section.

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

After all the above changes you just need:

$ make -j4 all

$ ./opticflow

In case you are running a system with a discrete card. I recommend to use the project Bumblebee. In this case:

$ optirun ./opticflow

LINKS OF INTEREST

To those who wants to start with developing in CUDA, I invite them to check this link:
https://github.com/Teknoman117/cuda 

BIBLIOGRAPHY

Horn, Berthold K., and Brian G. Schunck. "Determining optical flow." 1981 Technical Symposium East. International Society for Optics and Photonics, 1981.

Grewenig, S., Weickert, J., Schroers, C., & Bruhn, A. (2013). Cyclic schemes for PDE-based image analysis. Tech. Rep. 327, Department of Mathematics, Saarland University, Saarbrücken, Germany.

Grewenig, Sven, Joachim Weickert, and Andrés Bruhn. "From box filtering to fast explicit diffusion." Pattern Recognition. Springer Berlin Heidelberg, 2010. 533-542.

