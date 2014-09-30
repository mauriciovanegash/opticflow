//============================================================================
// Name        : opticflow.cpp
// Author      : Mauricio Vanegas
// Version     :
// Copyright   : Your copyright notice
// Description : Optic flow in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <limits.h>
using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "PDESolver.h"
#include "fed.h"

#define CV_BGR2GRAY	6

int main(void)
{
	PDESolver* solv = PDESolver::getInstance();
	int width = 640;
	int height = 480;

	// Open a webcamera
	cv::VideoCapture camera(0);
	cv::Mat         Bigframe;
	cv::Mat         frame;
	cv::Size		imSize = cv::Size(width,height);
//	unsigned char *cvMatTemp = solv->allocateSharedMemory<unsigned char>(width,height);

	// create CPU/GPU shared images - two for the images and one for the result
	cv::Mat imageT0(imSize, CV_8U, solv->allocateSharedMemory<unsigned char>(width, height));

	cv::Mat imageT1(imSize, CV_8U, solv->allocateSharedMemory<unsigned char>(width, height));

	cv::Mat u(imSize, CV_32F, solv->allocateSharedMemory<float>(width, height));

	cv::Mat v(imSize, CV_32F, solv->allocateSharedMemory<float>(width, height));

	if(!camera.isOpened()) return -1;

	cv::Rect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = width;
	roi.height = height;
	camera >> Bigframe;
	frame = Bigframe(roi);
//	cv::cvtColor(frame, imageT1, CV_BGR2GRAY);

	float *dev_imT0 = solv->allocateCudaMemory(width, height);
	float *dev_imT1 = solv->allocateCudaMemory(width, height);

	/*							FastJacobi							 */
	/* Initialise relaxation parameters for Fast-Jacobi with         */
	/* - cycle length:                      20                       */
	/* - number of cycles M:                5                        */
	/* - omega_max:                         1.0                      */

	// Vector of FJ Relaxation Parameters
	float *omega;
	// Cycle Length
	const int M = 5;
	//Number of cycles
	int N = fastjac_relax_params(20, 1.0f, 1, &omega);

	printf("Number of iterations %d\n", N);

	// smoothness
	// if image brightness is not within [0,1]
	// this parameter should be scaled appropriately
	const float alpha = 0.2f;

	// number of pyramid levels
	const int nLevels = M;//5;

	// number of solver iterations on each level
	int nSolverIters = N;//50;

	// number of warping iterations
	int nWarpIters = 1;//3

	// pI0 and pI1 will hold device pointers
	float **pI0 = new float *[nLevels];
	float **pI1 = new float *[nLevels];

	int *pW = new int [nLevels];
	int *pH = new int [nLevels];

	// device memory pointers
	float *d_tmp = solv->allocateCudaMemory(width, height);
	float *d_du0 = solv->allocateCudaMemory(width, height);
	float *d_dv0 = solv->allocateCudaMemory(width, height);
	float *d_du1 = solv->allocateCudaMemory(width, height);
	float *d_dv1 = solv->allocateCudaMemory(width, height);

	float *d_Ix = solv->allocateCudaMemory(width, height);
	float *d_Iy = solv->allocateCudaMemory(width, height);
	float *d_Iz = solv->allocateCudaMemory(width, height);

//	float *d_Ixx = solv->allocateCudaMemory(width, height);
//	float *d_Iyy = solv->allocateCudaMemory(width, height);
//	float *d_Ixy = solv->allocateCudaMemory(width, height);
//	float *d_Ixz = solv->allocateCudaMemory(width, height);
//	float *d_Iyz = solv->allocateCudaMemory(width, height);

	float *d_u = solv->allocateCudaMemory(width, height);
	float *d_v = solv->allocateCudaMemory(width, height);
	float *d_nu = solv->allocateCudaMemory(width, height);
	float *d_nv = solv->allocateCudaMemory(width, height);

	/* Allocating pyramid memory */
	int currentLevel = nLevels - 1;

	pW[currentLevel] = width;
	pH[currentLevel] = height;
	pI0[currentLevel] = dev_imT0;
	pI1[currentLevel] = dev_imT1;

	for (; currentLevel > 0; --currentLevel)
	{
		int nw = pW[currentLevel] / 2;
		int nh = pH[currentLevel] / 2;

		pI0[currentLevel-1] = solv->allocateCudaMemory(nw,nh);
		pI1[currentLevel-1] = solv->allocateCudaMemory(nw,nh);

		pW[currentLevel - 1] = nw;
		pH[currentLevel - 1] = nh;
	}

	int dataSize = AlignBuffer(width)*height;

	// Loop while capturing images
	while(1)
	{
		// Capture the image and store a gray conversion for the gpu
		camera >> Bigframe;
		frame = Bigframe(roi);
		imageT1.copyTo(imageT0);
		cv::cvtColor(frame, imageT1, CV_BGR2GRAY);

		solv->populateGPUBuffer(dev_imT0, (unsigned char*)imageT0.data, width, height, 255.0f);
		solv->populateGPUBuffer(dev_imT1, (unsigned char*)imageT1.data, width, height, 255.0f);

		// Perform GPU operations
		// prepare pyramid
		int currentLevel = nLevels - 1;

		for (; currentLevel > 0; --currentLevel)
		{
			solv->downSampling(pI0[currentLevel], pI0[currentLevel - 1], pW[currentLevel], pH[currentLevel]);
			solv->downSampling(pI1[currentLevel], pI1[currentLevel - 1], pW[currentLevel], pH[currentLevel]);
		}

		solv->setValueToGPUMem(d_u,0.0f,dataSize);
		solv->setValueToGPUMem(d_v,0.0f,dataSize);
		solv->setValueToGPUMem(d_nu,0.0f,dataSize);
		solv->setValueToGPUMem(d_nv,0.0f,dataSize);

		// compute flow
		for (; currentLevel < nLevels; ++currentLevel)
		{
			for (int warpIter = 0; warpIter < nWarpIters; ++warpIter)
			{
				solv->setValueToGPUMem(d_du0,0.0f,dataSize);
				solv->setValueToGPUMem(d_dv0,0.0f,dataSize);
				solv->setValueToGPUMem(d_du1,0.0f,dataSize);
				solv->setValueToGPUMem(d_dv1,0.0f,dataSize);

				// on current level we compute optical flow
				// between frame 0 and warped frame 1
				solv->warping(pI1[currentLevel],d_tmp,d_u,d_v,pW[currentLevel],pH[currentLevel]);

				solv->spatialImDerivative(pI0[currentLevel],d_tmp,d_Ix,d_Iy,d_Iz,pW[currentLevel],
						   					pH[currentLevel],CENTERED_FOUR_ORDER);
//				solv->spatialImDerivative(pI0[currentLevel],d_tmp,d_Ix,d_Iy,d_Iz,d_Ixx,d_Iyy,d_Ixy,d_Ixz,d_Iyz,
//											pW[currentLevel],pH[currentLevel]);

				for(int iter = 0; iter < nSolverIters; ++iter)
				{
					solv->jacobiIteration(d_u, d_v, d_du0, d_dv0, d_Ix, d_Iy, d_Iz, pW[currentLevel],
								   	   	   pH[currentLevel], omega[iter], alpha, d_du1, d_dv1);
					ptrSwap(d_du0, d_du1);
					ptrSwap(d_dv0, d_dv1);
				}

				// update u, v
				solv->vectorAdd(d_u, d_du0, pH[currentLevel] * AlignBuffer(pW[currentLevel]), d_u);
				solv->vectorAdd(d_v, d_dv0, pH[currentLevel] * AlignBuffer(pW[currentLevel]), d_v);
			}

			if (currentLevel != nLevels - 1)
			{
				// prolongate solution
				float scaleX = (float)pW[currentLevel + 1]/(float)pW[currentLevel];

				solv->upSampling(d_u, d_nu, scaleX, pW[currentLevel], pH[currentLevel]);

				float scaleY = (float)pH[currentLevel + 1]/(float)pH[currentLevel];

				solv->upSampling(d_v, d_nv, scaleY, pW[currentLevel], pH[currentLevel]);

				ptrSwap(d_u, d_nu);
				ptrSwap(d_v, d_nv);
			}
		}

//		/* Debugging */
//		int level = 4;
//		cv::Mat imTemp(cv::Size(pW[level],pH[level]), CV_8U, cvMatTemp);
//		solv->retrieveGPUBuffer((unsigned char*)imTemp.data, d_Iyz, pW[level], pH[level]);
//		cv::imshow("Debugging Pyramid", imTemp);

		solv->retrieveGPUBuffer((float*)u.data, d_u, width, height);
		solv->retrieveGPUBuffer((float*)v.data, d_v, width, height);

		//calculate angle and magnitude
		cv::Mat magnitude, angle;
		cv::cartToPolar(u, v, magnitude, angle, true);

		//translate magnitude to range [0;1]
		double vMax=50.0f;//,vMin=0.0f;
//		cv::Point pMin, pMax;
//		cv::minMaxLoc(magnitude,&vMin,&vMax,&pMin,&pMax);
		magnitude.convertTo(magnitude, -1, 1.0/vMax);

		//build hsv image
		cv::Mat _hsv[3], hsv;
		_hsv[0] = angle;
		_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
		_hsv[2] = magnitude;
		cv::merge(_hsv, 3, hsv);

		//convert to BGR and show
		cv::Mat bgr;//CV_32FC3 matrix
		cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
		cv::imshow("New Frame", imageT1);
		cv::imshow("Old Frame", imageT0);
		cv::imshow("Optical Flow", bgr);

		int c = cvWaitKey(10);
		if (27 == char(c))
			break;
	}

	// Exit
	solv->destroy();
	Bigframe.release();

	/* Example three */
//	FastJacobi();
//	FED();

	return EXIT_SUCCESS;
}
