/*
 * PDESolver.cpp
 *
 *  Created on: Jun 3, 2014
 *      Author: Mauricio Vanegas
 */

#include "PDESolver.h"

// Global static pointer used to ensure a single instance of the class.
PDESolver* PDESolver::k_Instance = NULL;

PDESolver::PDESolver()
{
	checkErrors(cuGPUInit());
}

PDESolver::~PDESolver()
{
	this->deallocateSharedMemory();
	this->deallocateCudaMemory();
}

void PDESolver::destroy(void)
{
	delete(k_Instance);
}

void PDESolver::getGPUDeviceInfo(void)
{
	checkErrors(cuInfoGPUDevice());
}

void PDESolver::callbackRegistration(void (*function)())
{
	this->setTimerExec(function);
}

void PDESolver::callbackRegistration(void (*function)(), int baseTime)
{
	this->setTimerExec(function,baseTime);
}

void PDESolver::spatialImDerivative(unsigned char *srcIm, unsigned char *tarIm,
									unsigned char *Ix, unsigned char *Iy, unsigned char *It,
									int width, int height, int type)
{
	checkErrors(cuImageDerivative(width, height, srcIm, tarIm, Ix, Iy, It, type));
}

void PDESolver::spatialImDerivative(float *srcIm, float *tarIm, float *Ix, float *Iy, float *It,
									int width, int height, int type)
{
	checkErrors(cuImageDerivative(width, height, srcIm, tarIm, Ix, Iy, It, type));
}

void PDESolver::spatialImDerivative(unsigned char *srcIm, unsigned char *tarIm,
							unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
							unsigned char *Ixx, unsigned char *Iyy, unsigned char *Ixy,
							unsigned char *Ixz, unsigned char *Iyz, int width, int height)
{
	checkErrors(cuImageDerivativeExtended(width, height, srcIm, tarIm, Ix, Iy, Iz, Ixx, Iyy, Ixy, Ixz, Iyz));
}

void PDESolver::spatialImDerivative(float *srcIm, float *tarIm, float *Ix, float *Iy, float *Iz,
						 float *Ixx, float *Iyy, float *Ixy, float *Ixz, float *Iyz, int width, int height)
{
	checkErrors(cuImageDerivativeExtended(width, height, srcIm, tarIm, Ix, Iy, Iz, Ixx, Iyy, Ixy, Ixz, Iyz));
}

void PDESolver::warping(unsigned char *srcIm, unsigned char *dstIm, float *dx, float *dy,
						int width, int height)
{
	checkErrors(cuWarping(width, height, srcIm, dstIm, dx, dy, false));
}

void PDESolver::warping(unsigned char *srcIm, unsigned char *dstIm, float *dx, float *dy,
						int width, int height, bool isTotalDisp)
{
	checkErrors(cuWarping(width, height, srcIm, dstIm, dx, dy, isTotalDisp));
}

void PDESolver::warping(float *srcIm, float *dstIm, float *dx, float *dy,
						int width, int height)
{
	checkErrors(cuWarping(width, height, srcIm, dstIm, dx, dy, false));
}

void PDESolver::warpingText(float *srcIm, float *dstIm, float *dx, float *dy,
							int width, int height)
{
	checkErrors(cuWarpingText(width, height, srcIm, dstIm, dx, dy));
}

void PDESolver::warping(float *srcIm, float *dstIm, float *dx, float *dy,
						int width, int height, bool isTotalDisp)
{
	checkErrors(cuWarping(width, height, srcIm, dstIm, dx, dy, isTotalDisp));
}

void PDESolver::downSampling(unsigned char *srcIm, unsigned char *dstIm, int width, int height)
{
	checkErrors(cuDownScale(width, height, srcIm, dstIm));
}

void PDESolver::downSampling(float *srcIm, float *dstIm, int width, int height)
{
	checkErrors(cuDownScale(width, height, srcIm, dstIm));
}

void PDESolver::upSampling(unsigned char *srcIm, unsigned char *dstIm, float scale, int width, int height)
{
	checkErrors(cuUpScale(width, height, scale, srcIm, dstIm));
//	checkErrors(cuUpScaleText(width, height, scale, srcIm, dstIm));
}

void PDESolver::upSampling(float *srcIm, float *dstIm, float scale, int width, int height)
{
	checkErrors(cuUpScale(width, height, scale, srcIm, dstIm));
}

void PDESolver::upSamplingTexture(float *srcIm, float *dstIm, float scale, int width, int height)
{
	checkErrors(cuUpScaleTexture(width, height, scale, srcIm, dstIm));
}

void PDESolver::vectorAdd(float *vec1, float *vec2, int length, float *sum)
{
	checkErrors(cuVectorAdd(vec1,vec2,length,sum));
}

void PDESolver::populateGPUBuffer(float *dst, unsigned char *src, int width, int height, float factor)
{
	checkErrors(cuCopyHostToGPU(width,height,src,dst,factor));
}

void PDESolver::populateGPUBuffer(float *dst, float *src, int width, int height, float factor)
{
	checkErrors(cuCopyHostToGPU(width,height,src,dst,factor));
}

void PDESolver::populateGPUBuffer(float *dst, unsigned char *src, int width, int height)
{
	checkErrors(cuCopyHostToGPU(width,height,src,dst,1.0f));
}

void PDESolver::populateGPUBuffer(float *dst, float *src, int width, int height)
{
	checkErrors(cuCopyHostToGPU(width,height,src,dst,1.0f));
}

void PDESolver::retrieveGPUBuffer(unsigned char *dst, float *src, int width, int height)
{
	checkErrors(cuCopyGPUtoHost(width,height,src,dst));
}

void PDESolver::retrieveGPUBuffer(float *dst, float *src, int width, int height)
{
	checkErrors(cuCopyGPUtoHost(width,height,src,dst));
}

void PDESolver::jacobiIteration(float *u, float *v, float *du0, float *dv0, float *Ix, float *Iy, float *Iz,
            					int iw, int ih, float omega, float alpha, float *du1, float *dv1)
{
	checkErrors(cuJacobiSolver(u, v, du0, dv0, Ix, Iy, Iz, iw, ih, omega, alpha, du1, dv1));
}

void PDESolver::sobelFunc(unsigned char *srcIm, unsigned char *dstIm, int width, int height)
{
	checkErrors(cuSobelFilter(width, height, srcIm, dstIm));
}

void PDESolver::sobelFunc(float *srcIm, float *dstIm, int width, int height)
{
	checkErrors(cuSobelFilter(width, height, srcIm, dstIm));
}

void PDESolver::blurredFilter(unsigned char *srcIm, unsigned char *dstIm, int width, int height)
{
	checkErrors(cuBoxFilter(width, height, srcIm, dstIm, 3, 3));
}

void PDESolver::blurredFilter(float *srcIm, float *dstIm, int width, int height)
{
	checkErrors(cuBoxFilter(width, height, srcIm, dstIm, 3, 3));
}

void PDESolver::distortionLutGen(float *dx, float *dy, int width, int height)
{
	checkErrors(cuDistortion(width, height, dx, dy));
}

void PDESolver::cartToPolarLutGen(float *dx, float *dy, int width, int height, int px, int py)
{
	checkErrors(cuCartToPolar(width, height, dx, dy, px, py));
}

void PDESolver::polarToCartLutGen(float *dx, float *dy, int width, int height, int px, int py)
{
	checkErrors(cuPolarToCart(width, height, dx, dy, px, py));
}

void PDESolver::setValueToGPUMem(unsigned char *ptr, unsigned char value, size_t size)
{
	checkErrors(cuGPUMemset(ptr,value,size));
}

void PDESolver::setValueToGPUMem(float *ptr, float value, size_t size)
{
	checkErrors(cuGPUMemset(ptr,value,size));
}

void PDESolver::memcopyHostToGPU(void *dst, void *src, size_t bytes)
{
	checkErrors(cuMemCopyHostToGPU(dst,src,bytes));
}

void PDESolver::memcopyGPUtoHost(void *dst, void *src, size_t bytes)
{
	checkErrors(cuMemCopyGPUtoHost(dst,src,bytes));
}

void PDESolver::deallocateSharedMemory(void)
{
	int cont = 0;
	for(std::vector<void*>::iterator it = alloc_host_memory.begin(); it != alloc_host_memory.end();++it)
	{
		checkErrors(cuDestroySharedBuffer(*it));
		++cont;
	}
	printf("There were deallocated %d pointers of pinned memory.\n",cont);
}

float* PDESolver::allocateCudaMemory(size_t width, size_t height)
{
	float* ptr = NULL;
	size_t bytes = AlignBuffer(width)*height;
	checkErrors(cuCreateGPUBuffer(bytes*sizeof(float), (void**)&ptr));

	this->alloc_cuda_memory.push_back(ptr);
	return ptr;
}

void PDESolver::deallocateCudaMemory(void)
{
	int cont = 0;
	for(std::vector<void*>::iterator it = alloc_cuda_memory.begin(); it != alloc_cuda_memory.end();++it)
	{
		checkErrors(cuDestroyGPUBuffer(*it));
		++cont;
	}
	printf("There were deallocated %d pointers of GPU memory.\n",cont);
}
