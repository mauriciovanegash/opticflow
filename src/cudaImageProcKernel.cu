/*
 * cudaImageProcKernel.cu
 *
 *  Created on: Jul 3, 2014
 *      Author: Mauricio Vanegas
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* CUDA includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cudaImageProcHeader.cuh"

int cuDistortion(int iw, int ih, float *dx, float *dy)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	float *dev_dx = dx, *dev_dy =dy;

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    distortion_kernel<<<blocks,threads>>>(iw, ih, is, dev_dx, dev_dy);
    int error = checkCudaErrors(cudaThreadSynchronize());

    return error;
}

int cuCartToPolar(int iw, int ih, float *dx, float *dy, int px, int py)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	float *dev_dx = dx, *dev_dy =dy;

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    cart2polar_kernel<<<blocks,threads>>>(iw, ih, is, dev_dx, dev_dy, px, py);
    int error = checkCudaErrors(cudaThreadSynchronize());

    return error;
}

int cuPolarToCart(int iw, int ih, float *dx, float *dy, int px, int py)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	float *dev_dx = dx, *dev_dy =dy;

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    polar2cart_kernel<<<blocks,threads>>>(iw, ih, is, dev_dx, dev_dy, px, py);
    int error = checkCudaErrors(cudaThreadSynchronize());

    return error;
}

int cuVectorAdd(float *vec1, float *vec2, int length, float *sum)
{
	float *dev_vec1, *dev_vec2;

	int error = checkCudaErrors(cuGetPointer((void**)&dev_vec1,vec1));
	error += checkCudaErrors(cuGetPointer((void**)&dev_vec2,vec2));

    dim3 threads(256);
    dim3 blocks(iDivUp(length, threads.x));

    add_kernel<<<blocks, threads>>>(dev_vec1, dev_vec2, length, sum);
    error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

int cuJacobiSolver(float *u0, float *v0, float *du0, float *dv0, float *Ix, float *Iy, float *Iz,
                    int iw, int ih, float omega, float alpha, float *du1, float *dv1)
{
	int is = AlignBuffer(iw);
	float *dev_u0, *dev_v0, *dev_du0, *dev_dv0, *dev_du1, *dev_dv1;

	int error = checkCudaErrors(cuGetPointer((void**)&dev_u0,u0));
	error += checkCudaErrors(cuGetPointer((void**)&dev_v0,v0));
	error += checkCudaErrors(cuGetPointer((void**)&dev_du0,du0));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dv0,dv0));
	error += checkCudaErrors(cuGetPointer((void**)&dev_du1,du1));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dv1,dv1));

    // CTA size
    dim3 threads(32, 6);
    // grid size
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    JacobiIteration<32,6><<<blocks, threads>>>(dev_u0, dev_v0, dev_du0, dev_dv0, Ix, Iy, Iz, iw, ih, is, omega,
    											alpha, dev_du1, dev_dv1);
    error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

/* cuCopyGPUtoHost overload */
int cuCopyGPUtoHost(int iw, int ih, float *source, unsigned char *dest)
{
	return cuCopyGPUtoHostPrototype<unsigned char>(iw,ih,source,dest);
}

int cuCopyGPUtoHost(int iw, int ih, float *source, float *dest)
{
	return cuCopyGPUtoHostPrototype<float>(iw,ih,source,dest);
}

/* cuCopyHostToGPU overload */
int cuCopyHostToGPU(int iw, int ih, unsigned char *source, float *dest, float factor)
{
	return cuCopyHostToGPUPrototype<unsigned char>(iw,ih,source,dest,factor);
}

int cuCopyHostToGPU(int iw, int ih, float *source, float *dest, float factor)
{
	return cuCopyHostToGPUPrototype<float>(iw,ih,source,dest,factor);
}

/* cuBoxFilter overload */
int cuBoxFilter(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh)
{
	return cuBoxFilterPrototype<unsigned char>(iw,ih,source,dest,bw,bh);
}

int cuBoxFilter(int iw, int ih, float *source, float *dest, int bw, int bh)
{
	return cuBoxFilterPrototype<float>(iw,ih,source,dest,bw,bh);
}

/* cuSobelFilter overload */
int cuSobelFilter(int iw, int ih, unsigned char *source, unsigned char *dest)
{
	return cuSobelFilterPrototype<unsigned char>(iw,ih,source,dest);
}

int cuSobelFilter(int iw, int ih, float *source, float *dest)
{
	return cuSobelFilterPrototype<float>(iw,ih,source,dest);
}

/* cuImageDerivative overload */
int cuImageDerivative(int iw, int ih, unsigned char *source, unsigned char *target,
								unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
								int type)
{
	return cuImageDerivativePrototype<unsigned char>(iw,ih,source,target,Ix,Iy,Iz,type);
}

int cuImageDerivative(int iw, int ih, float *source, float *target,
								float *Ix, float *Iy, float *Iz, int type)
{
	return cuImageDerivativePrototype<float>(iw,ih,source,target,Ix,Iy,Iz,type);
}

/* cuImageDerivativeExtended overload */
int cuImageDerivativeExtended(int iw, int ih, unsigned char *source, unsigned char *target,
								unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
								unsigned char *Ixx, unsigned char *Iyy, unsigned char *Ixy,
								unsigned char *Ixz, unsigned char *Iyz)
{
	return cuImageDerivativeExtendedPrototype<unsigned char>(iw,ih,source,target,Ix,Iy,Iz,Ixx,Iyy,Ixy,Ixz,Iyz);
}

int cuImageDerivativeExtended(int iw, int ih, float *source, float *target,
							  float *Ix, float *Iy, float *Iz, float *Ixx, float *Iyy, float *Ixy,
							  float *Ixz, float *Iyz)
{
	return cuImageDerivativeExtendedPrototype<float>(iw,ih,source,target,Ix,Iy,Iz,Ixx,Iyy,Ixy,Ixz,Iyz);
}

/* cuWarping overload */
int cuWarping(int iw, int ih, unsigned char *source, unsigned char *dest, float *dx, float *dy, bool isTotalDisp)
{
	return cuWarpingPrototype<unsigned char>(iw,ih,source,dest,dx,dy,isTotalDisp);
}

int cuWarping(int iw, int ih, float *source, float *dest, float *dx, float *dy, bool isTotalDisp)
{
	return cuWarpingPrototype<float>(iw,ih,source,dest,dx,dy,isTotalDisp);
}

int cuWarpingText(int iw, int ih, float *source, float *dest, float *dx, float *dy)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	float *dev_source, *dev_dest;

	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // mirror if a coordinate value is out-of-range
    texToWarp.addressMode[0] = cudaAddressModeMirror;
    texToWarp.addressMode[1] = cudaAddressModeMirror;
    texToWarp.filterMode = cudaFilterModeLinear;
    texToWarp.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    error += checkCudaErrors(cudaBindTexture2D(0, texToWarp, dev_source, iw, ih, is * sizeof(float)));

    // Execute the kernel
    warping_text_Kernel<<<blocks, threads>>>(iw, ih, is, dx, dy, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

/* cuDownScale overload */
int cuDownScale(int iw, int ih, unsigned char *source, unsigned char *dest)
{
	return cuDownScalePrototype<unsigned char>(iw,ih,source,dest);
}

int cuDownScale(int iw, int ih, float *source, float *dest)
{
	return cuDownScalePrototype<float>(iw,ih,source,dest);
}

/* cuDownScale overload */
int cuUpScale(int iw, int ih, float scale, unsigned char *source, unsigned char *dest)
{
	return cuUpScalePrototype<unsigned char>(iw,ih,scale,source,dest);
}

int cuUpScale(int iw, int ih, float scale, float *source, float *dest)
{
	return cuUpScalePrototype<float>(iw,ih,scale,source,dest);
}

int cuUpScaleTexture(int iw, int ih, float scale, float *source, float *dest)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw*2);
	int os = AlignBuffer(iw);
	float *dev_source, *dev_dest;
	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(32, 8);
    dim3 blocks(iDivUp(iw*2, threads.x), iDivUp(iw*2, threads.y));

    // mirror if a coordinate value is out-of-range
    texCoarse.addressMode[0] = cudaAddressModeMirror;
    texCoarse.addressMode[1] = cudaAddressModeMirror;
    texCoarse.filterMode = cudaFilterModeLinear;
    texCoarse.normalized = true;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

    error += checkCudaErrors(cudaBindTexture2D(0, texCoarse, dev_source, iw, ih, os * sizeof(float)));

    upscale_text_kernel<<<blocks, threads>>>(iw*2, ih*2, is, scale, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

/* cuGPUMemset overload */
int cuGPUMemset(unsigned char *ptr, unsigned char value, size_t bytes)
{
	int error = cuGPUMemsetPrototype<unsigned char>(bytes,ptr,value);

    return error;
}

int cuGPUMemset(float *ptr, float value, size_t size)
{
	int error = cuGPUMemsetPrototype<float>(size,ptr,value);

    return error;
}

int cuCreateSharedBuffer(size_t bytes, void **ptr)
{
	int error = checkCudaErrors(cudaHostAlloc(ptr, bytes, cudaHostAllocMapped));

    return error;
}

int cuDestroySharedBuffer(void* ptr)
{
	int error = checkCudaErrors(cudaFreeHost(ptr));

    return error;
}

int cuCreateGPUBuffer(size_t bytes, void **ptr)
{
	int error = checkCudaErrors(cudaMalloc(ptr, bytes));

    return error;
}

int cuMemCopyGPUtoHost(void *dst, void *src, size_t bytes)
{
	int error = checkCudaErrors(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost));

    return error;
}

int cuMemCopyHostToGPU(void *dst, void *src, size_t bytes)
{
	int error = checkCudaErrors(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice));

    return error;
}

int cuDestroyGPUBuffer(void* ptr)
{
	int error = checkCudaErrors(cudaFree(ptr));

	return error;
}

int cuGPUInit(void)
{
	int cuDevice = 0;

	int error = checkCudaErrors(cudaSetDevice(cuDevice));
	error += checkCudaErrors(cudaSetDeviceFlags(cudaDeviceMapHost));
	error += checkCudaErrors(cudaFree(0));

	return error;
}

int cuInfoGPUDevice(void)
{
	int cuVersion = 0;
	int cuDevice = 0;
	int cuDeviceCount = 0;
	cudaDeviceProp *properties = new(cudaDeviceProp);

	int	error = checkCudaErrors(cudaGetDeviceCount(&cuDeviceCount));
	error += checkCudaErrors(cudaDriverGetVersion(&cuVersion));
	error += checkCudaErrors(cudaGetDevice(&cuDevice));
	error += checkCudaErrors(cudaGetDeviceProperties(properties, cuDevice));
	if(error)
		return EXIT_FAILURE;

	printf("Number of GPU devices: %d\n", cuDeviceCount);
	printf("CUDA driver version: %d\n", cuVersion);
	printf("CUDA capability major/minor version number: %d.%d\n", properties->major, properties->minor);
	printf("Current device: %s\n", properties->name);
	printf("Global memory: %ld\n", properties->totalGlobalMem);
	printf("Maximum threads per block: %d\n", properties->maxThreadsPerBlock);
	printf("Maximum size per dimension:\n");
	printf("X %d Y %d Z %d\n", properties->maxThreadsDim[0], properties->maxThreadsDim[1],
								properties->maxThreadsDim[2]);
	printf("Maximum Grid size per dimension:\n");
	printf("X %d Y %d Z %d\n", properties->maxGridSize[0], properties->maxGridSize[1],
									properties->maxGridSize[2]);
	printf("Number of multiprocessors: %d\n", properties->multiProcessorCount);
	if(properties->concurrentKernels == 1)
		printf("This device can support multiple kernels in the same context simultaneously\n");
	else
		printf("This device cannot support multiple kernels in the same context simultaneously\n");

	printf("The compute mode is: %d\n", properties->computeMode);
	printf("Can the device map Host memory? (no=0, yes=1) %d\n", properties->canMapHostMemory);
	printf("Is the device integrated with the host memory system? (no=0, yes=1) %d\n", properties->integrated);

	return cudaSuccess;
}
