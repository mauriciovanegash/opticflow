/*
 * cudaImageProcHeader.cuh
 *
 *  Created on: 5 Aug 2014
 *      Author: Mauricio Vanegas
 */

#ifndef CUDAIMAGEPROCHEADER_CUH_
#define CUDAIMAGEPROCHEADER_CUH_

#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* CUDA includes */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <math_functions.h>

#define checkCudaErrors(val)	cuCheck( (val), #val, __FILE__, __LINE__ )
#define DEVICE_RESET 			cudaDeviceReset();
#define AlignBuffer(val)		iAlignUp( val, 32 )
#define pi						(4.0f*atan(1.0f))

texture<float, 2, cudaReadModeElementType> texToWarp;
texture<float, 2, cudaReadModeElementType> texCoarse;

template< typename T >
int cuCheck(T result, char const *const func, const char *const file, int const line)
{
    if(result)
    {
        fprintf(stderr, "  CUDA error at %s:%d code=%d(%s) \n  while calling \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        DEVICE_RESET
        // Make sure we call CUDA Device Reset before exiting
        //exit(EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    else
    	return EXIT_SUCCESS;
}

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m)
{
    return (n + m - 1) / m;
}

template< typename T >
__global__ void memset_kernel(int size, T *ptr, T value)
{
	// Calculate our pixel's location
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;

	for (int i = x; i < size; i += gridDim.x * blockDim.x)
	{
		ptr[i] = value;
	}
}

template< typename T >
__global__ void copyGPUtoHost_kernel(int iw, int ih, int is, float *source, T *dest)
{
	// Calculate our pixel's location
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(x >= 0 && x < iw && y >= 0 && y < ih)
	{
		dest[iw*y+x] = (T)source[is*y+x];
	}
}

template< typename T >
__global__ void copyHostToGPU_kernel(int iw, int ih, int is, T *source, float *dest, float factor)
{
	// Calculate our pixel's location
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if(x >= 0 && x < iw && y >= 0 && y < ih)
	{
		dest[is*y+x] = (float)source[iw*y+x]/factor;
	}
}

template< typename T >
__global__ void boxfilter_kernel(int iw, int ih, int is, T *source, T *dest, int bw, int bh)
{
    // Calculate our pixel's location
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Variables to store the sum
    int count = 0;
    float sum = 0.0;

    // Do the blur operation by summing the surround pixels
    for(int j = -(bh/2); j <= (bh/2); j++)
    {
        for(int i = -(bw/2); i <= (bw/2); i++)
        {
            // Verify that this offset is within the image boundaries
            if((x+i) < iw && (x+i) >= 0 && (y+j) < ih && (y+j) >= 0)
            {
                sum += (float) source[((y+j) * is) + (x+i)];
                count++;
            }
        }
    }

    // Average the sum
    sum /= (float) count;
    dest[(y * is) + x] = (T) sum;
}

template< typename T >
__global__ void sobelfilter_kernel(int iw, int ih, int is, T *source, T *dest)
{
    // Calculate our pixel's location
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Operate only if we are in the correct boundaries
    if(x > 0 && x < iw - 1 && y > 0 && y < ih - 1)
    {
        float gx = (float)(-source[is*(y-1)+(x-1)] + source[is*(y-1)+(x+1)] +
                   -2*source[is*(y)+(x-1)] + 2*source[is*(y)+(x+1)] +
                   -source[is*(y+1)+(x-1)] + source[is*(y+1)+(x+1)]);

        float gy = (float)(-source[is*(y-1)+(x-1)] - 2*source[is*(y-1)+(x)]
				   -source[is*(y-1)+(x+1)] +
                    source[is*(y+1)+(x-1)] + 2*source[is*(y+1)+(x)] +
                    source[is*(y+1)+(x+1)]);

        dest[is*y+x] = (T) sqrt(gx*gx + gy*gy);
    }
}

template< typename T >
__global__ void derivative_kernel(int iw, int ih, int is, T *Left, T *Right,
									T *Ix, T *Iy, T *Iz, int div,
									int i1, int i2, int i3, int i4, int i5)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    float tempL, tempR;

    // Operate only if we are in the correct boundaries
    if(x > 1 && x < iw - 2 && y > 1 && y < ih - 2)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(x-2)]);
		tempL += (float)(i2*Left[is*(y)+(x-1)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(x+1)]);
		tempL += (float)(i5*Left[is*(y)+(x+2)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(x-2)]);
		tempR += (float)(i2*Right[is*(y)+(x-1)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(x+1)]);
		tempR += (float)(i5*Right[is*(y)+(x+2)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(y-2)+(x)]);
		tempL += (float)(i2*Left[is*(y-1)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y+1)+(x)]);
		tempL += (float)(i5*Left[is*(y+2)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y-2)+(x)]);
		tempR += (float)(i2*Right[is*(y-1)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y+1)+(x)]);
		tempR += (float)(i5*Right[is*(y+2)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x <= 1 && y > 1 && y < ih - 2)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(2-x)]);
		tempL += (float)(i2*Left[is*(y)+(1-x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(x+1)]);
		tempL += (float)(i5*Left[is*(y)+(x+2)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(2-x)]);
		tempR += (float)(i2*Right[is*(y)+(1-x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(x+1)]);
		tempR += (float)(i5*Right[is*(y)+(x+2)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(y-2)+(x)]);
		tempL += (float)(i2*Left[is*(y-1)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y+1)+(x)]);
		tempL += (float)(i5*Left[is*(y+2)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y-2)+(x)]);
		tempR += (float)(i2*Right[is*(y-1)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y+1)+(x)]);
		tempR += (float)(i5*Right[is*(y+2)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x > 1 && x < iw - 2 && y <= 1)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(x-2)]);
		tempL += (float)(i2*Left[is*(y)+(x-1)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(x+1)]);
		tempL += (float)(i5*Left[is*(y)+(x+2)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(x-2)]);
		tempR += (float)(i2*Right[is*(y)+(x-1)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(x+1)]);
		tempR += (float)(i5*Right[is*(y)+(x+2)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(2-y)+(x)]);
		tempL += (float)(i2*Left[is*(1-y)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y+1)+(x)]);
		tempL += (float)(i5*Left[is*(y+2)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(2-y)+(x)]);
		tempR += (float)(i2*Right[is*(1-y)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y+1)+(x)]);
		tempR += (float)(i5*Right[is*(y+2)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x <= 1 && y <= 1)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(2-x)]);
		tempL += (float)(i2*Left[is*(y)+(1-x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(x+1)]);
		tempL += (float)(i5*Left[is*(y)+(x+2)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(2-x)]);
		tempR += (float)(i2*Right[is*(y)+(1-x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(x+1)]);
		tempR += (float)(i5*Right[is*(y)+(x+2)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(2-y)+(x)]);
		tempL += (float)(i2*Left[is*(1-y)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y+1)+(x)]);
		tempL += (float)(i5*Left[is*(y+2)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(2-y)+(x)]);
		tempR += (float)(i2*Right[is*(1-y)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y+1)+(x)]);
		tempR += (float)(i5*Right[is*(y+2)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x >= iw - 2 && x < iw && y > 1 && y < ih - 2)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(x-2)]);
		tempL += (float)(i2*Left[is*(y)+(x-1)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(2*iw-x-3)]);
		tempL += (float)(i5*Left[is*(y)+(2*iw-x-4)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(x-2)]);
		tempR += (float)(i2*Right[is*(y)+(x-1)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(2*iw-x-3)]);
		tempR += (float)(i5*Right[is*(y)+(2*iw-x-4)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(y-2)+(x)]);
		tempL += (float)(i2*Left[is*(y-1)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y+1)+(x)]);
		tempL += (float)(i5*Left[is*(y+2)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y-2)+(x)]);
		tempR += (float)(i2*Right[is*(y-1)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y+1)+(x)]);
		tempR += (float)(i5*Right[is*(y+2)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x >= iw - 2 && x < iw && y >= ih - 2 && y < ih)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(x-2)]);
		tempL += (float)(i2*Left[is*(y)+(x-1)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(2*iw-x-3)]);
		tempL += (float)(i5*Left[is*(y)+(2*iw-x-4)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(x-2)]);
		tempR += (float)(i2*Right[is*(y)+(x-1)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(2*iw-x-3)]);
		tempR += (float)(i5*Right[is*(y)+(2*iw-x-4)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(y-2)+(x)]);
		tempL += (float)(i2*Left[is*(y-1)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(2*ih-y-3)+(x)]);
		tempL += (float)(i5*Left[is*(2*ih-y-4)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y-2)+(x)]);
		tempR += (float)(i2*Right[is*(y-1)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(2*ih-y-3)+(x)]);
		tempR += (float)(i5*Right[is*(2*ih-y-4)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
    else if(x > 1 && x < iw - 2 && y >= ih - 2 && y < ih)
	{
		// x derivative
		tempL  = (float)(i1*Left[is*(y)+(x-2)]);
		tempL += (float)(i2*Left[is*(y)+(x-1)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(y)+(x+1)]);
		tempL += (float)(i5*Left[is*(y)+(x+2)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y)+(x-2)]);
		tempR += (float)(i2*Right[is*(y)+(x-1)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(y)+(x+1)]);
		tempR += (float)(i5*Right[is*(y)+(x+2)]);
		tempR /= (float)div;

		Ix[is*y+x] = (T)((tempL + tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];

		// y derivative
		tempL  = (float)(i1*Left[is*(y-2)+(x)]);
		tempL += (float)(i2*Left[is*(y-1)+(x)]);
		tempL += (float)(i3*Left[is*(y)+(x)]);
		tempL += (float)(i4*Left[is*(2*ih-y-3)+(x)]);
		tempL += (float)(i5*Left[is*(2*ih-y-4)+(x)]);
		tempL /= (float)div;

		tempR  = (float)(i1*Right[is*(y-2)+(x)]);
		tempR += (float)(i2*Right[is*(y-1)+(x)]);
		tempR += (float)(i3*Right[is*(y)+(x)]);
		tempR += (float)(i4*Right[is*(2*ih-y-3)+(x)]);
		tempR += (float)(i5*Right[is*(2*ih-y-4)+(x)]);
		tempR /= (float)div;

		Iy[is*y+x] = (T)((tempL + tempR)/2.0f);
	}
}

template< typename T >
__global__ void derivative_kernel_extended(int iw, int ih, int is, T *Left, T *Right,
											T *Ix, T *Iy, T *Iz, T *Ixx, T *Iyy, T *Ixy, T *Ixz, T *Iyz,
											int fd_div, int fd_i1, int fd_i2, int fd_i3, int fd_i4, int fd_i5,
											int sd_div, int sd_i1, int sd_i2, int sd_i3, int sd_i4, int sd_i5)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    float fd_tempL, fd_tempR, sd_tempL, sd_tempR;
    T tmpL, tmpR;

    // Operate only if we are in the correct boundaries
    if(x > 1 && x < iw - 2 && y > 1 && y < ih - 2)
	{
		// x derivative
    	tmpL = Left[is*(y)+(x-2)];
    	fd_tempL  = (float)(fd_i1*tmpL);
    	sd_tempL  = (float)(sd_i1*tmpL);
    	tmpL = Left[is*(y)+(x-1)];
    	fd_tempL += (float)(fd_i2*tmpL);
    	sd_tempL += (float)(sd_i2*tmpL);
    	tmpL = Left[is*(y)+(x)];
    	fd_tempL += (float)(fd_i3*tmpL);
    	sd_tempL += (float)(sd_i3*tmpL);
    	tmpL = Left[is*(y)+(x+1)];
    	fd_tempL += (float)(fd_i4*tmpL);
    	sd_tempL += (float)(sd_i4*tmpL);
    	tmpL = Left[is*(y)+(x+2)];
    	fd_tempL += (float)(fd_i5*tmpL);
    	sd_tempL += (float)(sd_i5*tmpL);
    	fd_tempL /= (float)fd_div;
    	sd_tempL /= (float)sd_div;

    	tmpR = Right[is*(y)+(x-2)];
    	fd_tempR  = (float)(fd_i1*tmpR);
    	sd_tempR  = (float)(sd_i1*tmpR);
    	tmpR = Right[is*(y)+(x-1)];
    	fd_tempR += (float)(fd_i2*tmpR);
    	sd_tempR += (float)(sd_i2*tmpR);
    	tmpR = Right[is*(y)+(x)];
    	fd_tempR += (float)(fd_i3*tmpR);
    	sd_tempR += (float)(sd_i3*tmpR);
    	tmpR = Right[is*(y)+(x+1)];
    	fd_tempR += (float)(fd_i4*tmpR);
    	sd_tempR += (float)(sd_i4*tmpR);
    	tmpR = Right[is*(y)+(x+2)];
    	fd_tempR += (float)(fd_i5*tmpR);
    	sd_tempR += (float)(sd_i5*tmpR);
    	fd_tempR /= (float)fd_div;
    	sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(y-2)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y-1)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y+1)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y+2)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y-2)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y-1)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y+1)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y+2)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x <= 1 && y > 1 && y < ih - 2)
	{
		// x derivative
		tmpL = Left[is*(y)+(2-x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(1-x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(x+1)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(x+2)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(2-x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(1-x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(x+1)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(x+2)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(y-2)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y-1)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y+1)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y+2)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y-2)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y-1)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y+1)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y+2)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x > 1 && x < iw - 2 && y <= 1)
	{
		// x derivative
		tmpL = Left[is*(y)+(x-2)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(x-1)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(x+1)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(x+2)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(x-2)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(x-1)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(x+1)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(x+2)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(2-y)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(1-y)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y+1)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y+2)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(2-y)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(1-y)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y+1)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y+2)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x <= 1 && y <= 1)
	{
		// x derivative
		tmpL = Left[is*(y)+(2-x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(1-x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(x+1)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(x+2)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(2-x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(1-x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(x+1)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(x+2)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(2-y)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(1-y)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y+1)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y+2)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(2-y)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(1-y)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y+1)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y+2)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x >= iw - 2 && x < iw && y > 1 && y < ih - 2)
	{
		// x derivative
		tmpL = Left[is*(y)+(x-2)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(x-1)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(2*iw-x-3)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(2*iw-x-4)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(x-2)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(x-1)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(2*iw-x-3)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(2*iw-x-4)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(y-2)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y-1)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y+1)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y+2)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y-2)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y-1)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y+1)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y+2)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x >= iw - 2 && x < iw && y >= ih - 2 && y < ih)
	{
		// x derivative
		tmpL = Left[is*(y)+(x-2)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(x-1)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(2*iw-x-3)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(2*iw-x-4)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(x-2)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(x-1)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(2*iw-x-3)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(2*iw-x-4)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(y-2)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y-1)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(2*ih-y-3)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(2*ih-y-4)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y-2)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y-1)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(2*ih-y-3)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(2*ih-y-4)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
    else if(x > 1 && x < iw - 2 && y >= ih - 2 && y < ih)
	{
		// x derivative
		tmpL = Left[is*(y)+(x-2)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y)+(x-1)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(y)+(x+1)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(y)+(x+2)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y)+(x-2)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y)+(x-1)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(y)+(x+1)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(y)+(x+2)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Ix[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Ixx[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iz[is*y+x] = Right[is*y+x] - Left[is*y+x];
		Ixz[is*y+x] = (T)(fd_tempR - fd_tempL);

		// y derivative
		tmpL = Left[is*(y-2)+(x)];
		fd_tempL  = (float)(fd_i1*tmpL);
		sd_tempL  = (float)(sd_i1*tmpL);
		tmpL = Left[is*(y-1)+(x)];
		fd_tempL += (float)(fd_i2*tmpL);
		sd_tempL += (float)(sd_i2*tmpL);
		tmpL = Left[is*(y)+(x)];
		fd_tempL += (float)(fd_i3*tmpL);
		sd_tempL += (float)(sd_i3*tmpL);
		tmpL = Left[is*(2*ih-y-3)+(x)];
		fd_tempL += (float)(fd_i4*tmpL);
		sd_tempL += (float)(sd_i4*tmpL);
		tmpL = Left[is*(2*ih-y-4)+(x)];
		fd_tempL += (float)(fd_i5*tmpL);
		sd_tempL += (float)(sd_i5*tmpL);
		fd_tempL /= (float)fd_div;
		sd_tempL /= (float)sd_div;

		tmpR = Right[is*(y-2)+(x)];
		fd_tempR  = (float)(fd_i1*tmpR);
		sd_tempR  = (float)(sd_i1*tmpR);
		tmpR = Right[is*(y-1)+(x)];
		fd_tempR += (float)(fd_i2*tmpR);
		sd_tempR += (float)(sd_i2*tmpR);
		tmpR = Right[is*(y)+(x)];
		fd_tempR += (float)(fd_i3*tmpR);
		sd_tempR += (float)(sd_i3*tmpR);
		tmpR = Right[is*(2*ih-y-3)+(x)];
		fd_tempR += (float)(fd_i4*tmpR);
		sd_tempR += (float)(sd_i4*tmpR);
		tmpR = Right[is*(2*ih-y-4)+(x)];
		fd_tempR += (float)(fd_i5*tmpR);
		sd_tempR += (float)(sd_i5*tmpR);
		fd_tempR /= (float)fd_div;
		sd_tempR /= (float)sd_div;

		Iy[is*y+x] = (T)((fd_tempL + fd_tempR)/2.0f);
		Iyy[is*y+x] = (T)((sd_tempL + sd_tempR)/2.0f);

		Iyz[is*y+x] = (T)(fd_tempR - fd_tempL);
	}
}

__global__ void cart2polar_kernel(int iw, int ih, int is, float *dx, float *dy, int px, int py)
{
    // Calculate our pixel's location
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	const int indexing = is*y+x;

	const float theta = (2.0f*((float)x/(float)iw)-1.0f)*pi;
	const float r = (float)y/2.0f;
	float Xpole = (float)px;
	float Ypole = (float)py;

	if(x >= 0  && x < iw && y >= 0 && y < ih)
	{
		dx[indexing] = r*cos(theta)+Xpole-x;
		dy[indexing] = r*sin(theta)+Ypole-y;
	}
}

__global__ void polar2cart_kernel(int iw, int ih, int is, float *dx, float *dy, int px, int py)
{
    // Calculate our pixel's location
	const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	float Xpole = (float)px;
	float Ypole = (float)py;
	const float theta = (2.0f*((float)x/(float)iw)-1.0f)*pi;
	const float r = (float)y/2.0f;
	const float theta_grid = atan2((float)(y-ih/2),(float)(x-iw/2));
	const float r_grid = hypot((float)(x-iw/2),(float)(y-ih/2));
	float tx = r*cos(theta)+Xpole;
	float ty = r*sin(theta)+Ypole;
	int nx = (int)floor(tx);
	int ny = (int)floor(ty);

	if(x >= 0 && x < iw && y >= 0 && y < ih)
	{
		dx[is*ny+nx] = (float)x - tx;
		dy[is*ny+nx] = (float)y - ty;
		if(fabs(tx-floor(tx))<0.5f && fabs(ty-floor(ty))<0.5f)
		{
			dx[is*ny+nx+1] = (float)(x-1) - tx;
			dy[is*ny+nx+1] = (float)(y) - ty;
			dx[is*(ny+1)+nx] = (float)(x) - tx;
			dy[is*(ny+1)+nx] = (float)(y-1) - ty;
			dx[is*(ny+1)+nx+1] = (float)(x-1) - tx;
			dy[is*(ny+1)+nx+1] = (float)(y-1) - ty;
		}
		else
		{
			dx[is*ny+nx+1] = (float)(x+1) - tx;
			dy[is*ny+nx+1] = (float)(y) - ty;
			dx[is*(ny+1)+nx] = (float)(x) - tx;
			dy[is*(ny+1)+nx] = (float)(y+1) - ty;
			dx[is*(ny+1)+nx+1] = (float)(x+1) - tx;
			dy[is*(ny+1)+nx+1] = (float)(y+1) - ty;
		}
	}
}

__global__ void distortion_kernel(int iw, int ih, int is, float *dx, float *dy)
{
    // Calculate our pixel's location
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    const float theta = 45.0f*pi/180.0f;
    float distance = 2;

    if(x >= 0  && x < iw && y >= 0 && y < ih)
    {
		dx[is*y+x] = ((float)x-iw/2)*cos(distance*theta)-((float)y-ih/2)*sin(distance*theta);

		dy[is*y+x] = ((float)x-iw/2)*sin(distance*theta)+((float)y-ih/2)*cos(distance*theta);
    }
}

__global__ void warping_text_Kernel(int iw, int ih, int is,
                              float *dx, float *dy, float *dest)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int pos = x + y * is;

    if (x >= iw || y >= ih) return;

    float xDisp = ((float)x + dx[pos] + 0.5f) / (float)iw;
    float yDisp = ((float)y + dy[pos] + 0.5f) / (float)ih;

    dest[pos] = tex2D(texToWarp, xDisp, yDisp);
}

template< typename T >
__global__ void warping_kernel(int iw, int ih, int is, T *source, T *dest,
								float *dx, float *dy, bool isTotalDisp)
{
    // Calculate our pixel's location
    const int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    /* Setting the total displacement */
    float xTotDisp = floor(dx[is*y+x]);
    float yTotDisp = floor(dy[is*y+x]);

    /* Setting the partial displacement according to pixel position */
    int xw = (int)xTotDisp;
    int yw = (int)yTotDisp;
    int xwAbs = (int)fabs(xTotDisp);
    int ywAbs = (int)fabs(yTotDisp);

    /* Defining the decimal fractions for displacement */
    float xDec = dx[is*y+x] - xTotDisp;
    float xDecCom = 1-xDec;
    float yDec = dy[is*y+x] - yTotDisp;
    float yDecCom = 1-yDec;

    /* Defining the weights for bilinear interpolation */
    float w00 = xDecCom*yDecCom;
    float w10 = xDec*yDecCom;
    float w01 = xDecCom*yDec;
    float w11 = xDec*yDec;

    // Operate only if we are in the correct boundaries
    if(!isTotalDisp && (xwAbs<iw) && (ywAbs<ih) && (x<iw) && (y<ih))
    {
    	if(x+xw >= 0 && x+xw < iw-1 && y+yw >= 0 && y+yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(y+yw))+x+xw] + w10*source[(is*(y+yw))+x+xw+1] +
							   w01*source[(is*(y+yw+1))+x+xw] + w11*source[(is*(y+yw+1))+x+xw+1]);
		}
    	else if(x+xw >= iw-1 && y+yw >= 0 && y+yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(y+yw))+2*iw-x-xw-2] + w10*source[(is*(y+yw))+2*iw-x-xw-3] +
							   w01*source[(is*(y+yw+1))+2*iw-x-xw-2] + w11*source[(is*(y+yw+1))+2*iw-x-xw-3]);
		}
    	else if(x+xw >= 0 && x+xw < iw-1 && y+yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-y-yw-2))+x+xw] + w10*source[(is*(2*ih-y-yw-2))+x+xw+1] +
							   w01*source[(is*(2*ih-y-yw-3))+x+xw] + w11*source[(is*(2*ih-y-yw-3))+x+xw+1]);
		}
    	else if(x+xw < 0 && y+yw >= 0 && y+yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(y+yw))-x-xw] + w10*source[(is*(y+yw))-x-xw-1] +
							   w01*source[(is*(y+yw+1))-x-xw] + w11*source[(is*(y+yw+1))-x-xw-1]);
		}
		else if(x+xw >= 0 && x+xw < iw-1 && y+yw < 0)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-y-yw))+x+xw] + w10*source[(is*(-y-yw))+x+xw+1] +
							   w01*source[(is*(-y-yw-1))+x+xw] + w11*source[(is*(-y-yw-1))+x+xw+1]);
		}
		else if(x+xw < 0 && y+yw < 0)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-y-yw))-x-xw] + w10*source[(is*(-y-yw))-x-xw-1] +
							   w01*source[(is*(-y-yw-1))-x-xw] + w11*source[(is*(-y-yw-1))-x-xw-1]);
		}
		else if(x+xw >= iw-1 && y+yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-y-yw-2))+2*iw-x-xw-2] + w10*source[(is*(2*ih-y-yw-2))+2*iw-x-xw-3] +
							   w01*source[(is*(2*ih-y-yw-3))+2*iw-x-xw-2] + w11*source[(is*(2*ih-y-yw-3))+2*iw-x-xw-3]);
		}
		else if(x+xw < 0 && y+yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-y-yw-2))-x-xw] + w10*source[(is*(2*ih-y-yw-2))-x-xw-1] +
							   w01*source[(is*(2*ih-y-yw-3))-x-xw] + w11*source[(is*(2*ih-y-yw-3))-x-xw-1]);
		}
		else if(x+xw >= iw-1 && y+yw < 0)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-y-yw))+2*iw-x-xw-2] + w10*source[(is*(-y-yw))+2*iw-x-xw-3] +
							   w01*source[(is*(-y-yw-1))+2*iw-x-xw-2] + w11*source[(is*(-y-yw-1))+2*iw-x-xw-3]);
		}
    }
    else if(isTotalDisp && (xwAbs<2*iw) && (ywAbs<2*ih) && (x<iw) && (y<ih))
    {
    	if(xw >= 0 && xw < iw-1 && yw >= 0 && yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(yw))+xw] + w10*source[(is*(yw))+xw+1] +
							   w01*source[(is*(yw+1))+xw] + w11*source[(is*(yw+1))+xw+1]);
		}
		else if(xw >= iw-1 && yw >= 0 && yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(yw))+2*iw-xw-2] + w10*source[(is*(yw))+2*iw-xw-3] +
							   w01*source[(is*(yw+1))+2*iw-xw-2] + w11*source[(is*(yw+1))+2*iw-xw-3]);
		}
		else if(xw >= 0 && xw < iw-1 && yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-yw-2))+xw] + w10*source[(is*(2*ih-yw-2))+xw+1] +
							   w01*source[(is*(2*ih-yw-3))+xw] + w11*source[(is*(2*ih-yw-3))+xw+1]);
		}
		else if(xw < 0 && xw >= -iw+1 && yw >= 0 && yw < ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(yw))-xw] + w10*source[(is*(yw))-xw-1] +
							   w01*source[(is*(yw+1))-xw] + w11*source[(is*(yw+1))-xw-1]);
		}
		else if(xw >= 0 && xw < iw-1 && yw < 0 && yw >= -ih+1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-yw))+xw] + w10*source[(is*(-yw))+xw+1] +
							   w01*source[(is*(-yw-1))+xw] + w11*source[(is*(-yw-1))+xw+1]);
		}
		else if(xw < 0 && xw >= -iw+1 && yw < 0 && yw >= -ih+1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-yw))-xw] + w10*source[(is*(-yw))-xw-1] +
							   w01*source[(is*(-yw-1))-xw] + w11*source[(is*(-yw-1))-xw-1]);
		}
		else if(xw >= iw-1 && yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-yw-2))+2*iw-xw-2] + w10*source[(is*(2*ih-yw-2))+2*iw-xw-3] +
							   w01*source[(is*(2*ih-yw-3))+2*iw-xw-2] + w11*source[(is*(2*ih-yw-3))+2*iw-xw-3]);
		}
		else if(xw < 0 && xw >= -iw+1 && yw >= ih-1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(2*ih-yw-2))-xw] + w10*source[(is*(2*ih-yw-2))-xw-1] +
							   w01*source[(is*(2*ih-yw-3))-xw] + w11*source[(is*(2*ih-yw-3))-xw-1]);
		}
		else if(xw >= iw-1 && yw < 0 && yw >= -ih+1)
		{
			dest[is*y+x] = (T)(w00*source[(is*(-yw))+2*iw-xw-2] + w10*source[(is*(-yw))+2*iw-xw-3] +
							   w01*source[(is*(-yw-1))+2*iw-xw-2] + w11*source[(is*(-yw-1))+2*iw-xw-3]);
		}
    }
    else
	{
		dest[is*y+x] = (T)0;
	}
}

__global__ void add_kernel(float *vec1, float *vec2, int length, float *sum)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= length) return;

    sum[pos] = vec1[pos] + vec2[pos];
}

template< typename T >
__global__ void downscale_kernel(int iw, int ih, int is, int os, T *source, T *dest)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int ox = x/2;
	const int oy = y/2;

    if(x > 0 && x < iw-1 && y > 0 && y < ih-1 && !(x%2) && !(y%2))
    {
    	dest[os*oy+ox] = (T)(0.25f * (source[is*y+x-1] + source[is*y+x+1] + source[is*(y+1)+x] + source[is*(y-1)+x]));
    }
    else if(x == 0 && y > 0 && y < ih - 1 && !(y%2))
    {
    	dest[os*oy+ox] = (T)(0.25f * (source[is*y+x+1] + source[is*y+x+1] + source[is*(y+1)+x] + source[is*(y-1)+x]));
    }
    else if(x > 0 && x < iw - 1 && y == 0 && !(x%2))
	{
		dest[os*oy+ox] = (T)(0.25f * (source[is*y+x-1] + source[is*y+x+1] + source[is*(y+1)+x] + source[is*(y+1)+x]));
	}
}

template< typename T >
__global__ void upscale_kernel(int iw, int ih, int is, int os, float scale, T *source, T *dest)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    const int ox = x/2;
    const int oy = y/2;

    if(x < iw-2 && y < ih-2 && !(x%2) && !(y%2))
    {
		// scale interpolated vector to match next pyramid level resolution
    	dest[is*y+x] = (T)(scale*(source[(os*oy)+ox]));
    	dest[is*y+x+1] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy)+ox+1]));
		dest[is*(y+1)+x] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy+1)+ox]));
		dest[is*(y+1)+x+1] = (T)(scale*0.25f*(source[os*(oy)+ox] + source[os*(oy)+ox+1] +
											  source[os*(oy+1)+ox] + source[os*(oy+1)+ox+1]));
	}
    else if(x == iw-2 && y < ih-2 && !(y%2))
	{
    	// scale interpolated vector to match next pyramid level resolution
		dest[is*y+x] = (T)(scale*(source[(os*oy)+ox]));
		dest[is*y+x+1] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy)+ox-1]));
		dest[is*(y+1)+x] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy+1)+ox]));
		dest[is*(y+1)+x+1] = (T)(scale*0.25f*(source[os*(oy)+ox] + source[os*(oy)+ox-1] +
											  source[os*(oy+1)+ox] + source[os*(oy+1)+ox-1]));
	}
	else if(x < iw-2 && y == ih-2 && !(x%2))
	{
		// scale interpolated vector to match next pyramid level resolution
		dest[is*y+x] = (T)(scale*(source[(os*oy)+ox]));
		dest[is*y+x+1] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy)+ox+1]));
		dest[is*(y+1)+x] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy-1)+ox]));
		dest[is*(y+1)+x+1] = (T)(scale*0.25f*(source[os*(oy)+ox] + source[os*(oy)+ox+1] +
											  source[os*(oy-1)+ox] + source[os*(oy-1)+ox+1]));
	}
	else if(x == iw-2 && y == ih-2)
	{
		// scale interpolated vector to match next pyramid level resolution
		dest[is*y+x] = (T)(scale*(source[(os*oy)+ox]));
		dest[is*y+x+1] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy)+ox-1]));
		dest[is*(y+1)+x] = (T)(scale*0.5f*(source[os*(oy)+ox] + source[os*(oy-1)+ox]));
		dest[is*(y+1)+x+1] = (T)(scale*0.25f*(source[os*(oy)+ox] + source[os*(oy)+ox-1] +
											  source[os*(oy-1)+ox] + source[os*(oy-1)+ox-1]));
	}
}

__global__ void upscale_text_kernel(int iw, int ih, int is, float scale, float *dest)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= iw || y >= ih) return;

    float xDisp = ((float)x + 0.5f) / (float)iw;
    float yDisp = ((float)y + 0.5f) / (float)ih;

    // exploit hardware interpolation
    // and scale interpolated vector to match next pyramid level resolution
    dest[x + y * is] = tex2D(texCoarse, xDisp, yDisp) * scale;
}

template<int bx, int by>
__global__ void JacobiIteration(float *u0, float *v0, float *du0, float *dv0, float *Ix, float *Iy, float *Iz,
                     int w, int h, int s, float omega, float alpha, float *du1, float *dv1)
{
    volatile __shared__ float du[(bx + 2) * (by + 2)];
    volatile __shared__ float dv[(bx + 2) * (by + 2)];
    volatile __shared__ float u[(bx + 2) * (by + 2)];
	volatile __shared__ float v[(bx + 2) * (by + 2)];

    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;

    // position within global memory array
    const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

    // position within shared memory array
    const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

    // Load data to shared memory.
    // load tile being processed
    du[shMemPos] = du0[pos];
    dv[shMemPos] = dv0[pos];
    u[shMemPos] = u0[pos];
	v[shMemPos] = v0[pos];

    // load necessary neighbouring elements
    // We clamp out-of-range coordinates.
    // It is equivalent to mirroring
    // because we access data only one step away from borders.
    if (threadIdx.y == 0)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        x = min(bsx + threadIdx.x, w - 1);
        // row just below the tile
        y = max(bsy - 1, 0);
        gmPos = y * s + x;
        smPos = threadIdx.x + 1;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
        u[smPos] = u0[gmPos];
		v[smPos] = v0[gmPos];

        // row above the tile
        y = min(bsy + by, h - 1);
        smPos += (by + 1) * (bx + 2);
        gmPos  = y * s + x;
        du[smPos] = du0[gmPos];
        dv[smPos] = dv0[gmPos];
        u[smPos] = u0[gmPos];
		v[smPos] = v0[gmPos];
    }
    else if (threadIdx.y == 1)
    {
        // beginning of the tile
        const int bsx = blockIdx.x * blockDim.x;
        const int bsy = blockIdx.y * blockDim.y;
        // element position within matrix
        int x, y;
        // element position within linear array
        // gm - global memory
        // sm - shared memory
        int gmPos, smPos;

        y = min(bsy + threadIdx.x, h - 1);
        // column to the left
        x = max(bsx - 1, 0);
        smPos = bx + 2 + threadIdx.x * (bx + 2);
        gmPos = x + y * s;

        // check if we are within tile
        if (threadIdx.x < by)
        {
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            u[smPos] = u0[gmPos];
			v[smPos] = v0[gmPos];
            // column to the right
            x = min(bsx + bx, w - 1);
            gmPos  = y * s + x;
            smPos += bx + 1;
            du[smPos] = du0[gmPos];
            dv[smPos] = dv0[gmPos];
            u[smPos] = u0[gmPos];
			v[smPos] = v0[gmPos];
        }
    }

    __syncthreads();

    if (ix >= w || iy >= h) return;

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + bx + 2;
    down  = shMemPos - bx - 2;

    float sumdU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
    float sumdV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;
    float sumU = (u[left] + u[right] + u[up] + u[down]) * 0.25f;
	float sumV = (v[left] + v[right] + v[up] + v[down]) * 0.25f;

    float frac = (Ix[pos] * sumdU + Iy[pos] * sumdV + Iz[pos])
                 / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = du[shMemPos] * (1.0f-omega) + omega * (sumU - u[shMemPos] + sumdU - Ix[pos] * frac);
    dv1[pos] = dv[shMemPos] * (1.0f-omega) + omega * (sumV - v[shMemPos] + sumdV - Iy[pos] * frac);
}

template<int bx, int by>
__global__ void JacobiIterationExtended(float *u0, float *v0, float *du0, float *dv0, float *Ix, float *Iy,
				float *Iz, float *Ixx, float *Iyy, float *Ixy, float *Ixz, float *Iyz,
				int w, int h, int s, float omega, float alpha, float beta, float *du1, float *dv1)
{
	volatile __shared__ float du[(bx + 2) * (by + 2)];
	volatile __shared__ float dv[(bx + 2) * (by + 2)];
	volatile __shared__ float u[(bx + 2) * (by + 2)];
	volatile __shared__ float v[(bx + 2) * (by + 2)];

	const int ix = threadIdx.x + blockIdx.x * blockDim.x;
	const int iy = threadIdx.y + blockIdx.y * blockDim.y;

	// position within global memory array
	const int pos = min(ix, w - 1) + min(iy, h - 1) * s;

	// position within shared memory array
	const int shMemPos = threadIdx.x + 1 + (threadIdx.y + 1) * (bx + 2);

	// Load data to shared memory.
	// load tile being processed
	du[shMemPos] = du0[pos];
	dv[shMemPos] = dv0[pos];
	u[shMemPos] = u0[pos];
	v[shMemPos] = v0[pos];

	// load necessary neighbouring elements
	// We clamp out-of-range coordinates.
	// It is equivalent to mirroring
	// because we access data only one step away from borders.
	if (threadIdx.y == 0)
	{
		// beginning of the tile
		const int bsx = blockIdx.x * blockDim.x;
		const int bsy = blockIdx.y * blockDim.y;
		// element position within matrix
		int x, y;
		// element position within linear array
		// gm - global memory
		// sm - shared memory
		int gmPos, smPos;

		x = min(bsx + threadIdx.x, w - 1);
		// row just below the tile
		y = max(bsy - 1, 0);
		gmPos = y * s + x;
		smPos = threadIdx.x + 1;
		du[smPos] = du0[gmPos];
		dv[smPos] = dv0[gmPos];
		u[smPos] = u0[gmPos];
		v[smPos] = v0[gmPos];

		// row above the tile
		y = min(bsy + by, h - 1);
		smPos += (by + 1) * (bx + 2);
		gmPos  = y * s + x;
		du[smPos] = du0[gmPos];
		dv[smPos] = dv0[gmPos];
		u[smPos] = u0[gmPos];
		v[smPos] = v0[gmPos];
	}
	else if (threadIdx.y == 1)
	{
		// beginning of the tile
		const int bsx = blockIdx.x * blockDim.x;
		const int bsy = blockIdx.y * blockDim.y;
		// element position within matrix
		int x, y;
		// element position within linear array
		// gm - global memory
		// sm - shared memory
		int gmPos, smPos;

		y = min(bsy + threadIdx.x, h - 1);
		// column to the left
		x = max(bsx - 1, 0);
		smPos = bx + 2 + threadIdx.x * (bx + 2);
		gmPos = x + y * s;

		// check if we are within tile
		if (threadIdx.x < by)
		{
			du[smPos] = du0[gmPos];
			dv[smPos] = dv0[gmPos];
			u[smPos] = u0[gmPos];
			v[smPos] = v0[gmPos];
			// column to the right
			x = min(bsx + bx, w - 1);
			gmPos  = y * s + x;
			smPos += bx + 1;
			du[smPos] = du0[gmPos];
			dv[smPos] = dv0[gmPos];
			u[smPos] = u0[gmPos];
			v[smPos] = v0[gmPos];
		}
	}

    __syncthreads();

    if (ix >= w || iy >= h) return;

    // now all necessary data are loaded to shared memory
    int left, right, up, down;
    left  = shMemPos - 1;
    right = shMemPos + 1;
    up    = shMemPos + bx + 2;
    down  = shMemPos - bx - 2;

    float sumdU = (du[left] + du[right] + du[up] + du[down]) * 0.25f;
	float sumdV = (dv[left] + dv[right] + dv[up] + dv[down]) * 0.25f;
	float sumU = (u[left] + u[right] + u[up] + u[down]) * 0.25f;
	float sumV = (v[left] + v[right] + v[up] + v[down]) * 0.25f;

    /* Constants for robust functions */
    float d_epsilon = 0.001f;
    float s_epsilon = 0.001f;
    float temp1 = 0.0f;
    float temp2 = 0.0f;

    temp1 = (Ix*du[shMemPos]+Iy*dv[shMemPos]+Iz);
    float robustFuncData1 = 1.0f/sqrt(temp1 * temp1 + d_epsilon * d_epsilon);
    temp1 = (Ixx*du[shMemPos]+Ixy*dv[shMemPos]+Ixz);
    temp2 = (Ixy*du[shMemPos]+Iyy*dv[shMemPos]+Iyz);
    float robustFuncData2 = 1.0f/sqrt(temp1 * temp1 + temp2 * temp2 + d_epsilon * d_epsilon);

    float gradUdu = (u[right] - u[left] + du[right] - du[left])/omega;
    float gradVdv = (v[right] - v[left] + dv[right] - dv[left])/omega;
    float robustFuncSmooth = 1.0f/sqrt(gradUdu * gradUdu + gradVdv * gradVdv + s_epsilon * s_epsilon);

    float frac = (Ix[pos] * sumU + Iy[pos] * sumV + Iz[pos])
                 / (Ix[pos] * Ix[pos] + Iy[pos] * Iy[pos] + alpha);

    du1[pos] = du[shMemPos] * (1.0f-omega) + omega * (sumU - Ix[pos] * frac);
	dv1[pos] = dv[shMemPos] * (1.0f-omega) + omega * (sumV - Iy[pos] * frac);
}

/* This function return the appropriate pointer regardless the type of	*
 * memory, GPU/Host unified memory or device memory. 					*/
template< typename T >
inline cudaError_t cuGetPointer(void **dev_ptr, T *ptr)
{
	cudaPointerAttributes ptrAttributes;
	cudaError_t error = cudaSuccess;

	cudaPointerGetAttributes(&ptrAttributes,ptr);
	if(ptrAttributes.memoryType == cudaMemoryTypeDevice)
		*dev_ptr = ptr;
	else
		error = cudaHostGetDevicePointer(dev_ptr, (void*)ptr, 0);

	return error;
}

template< typename T >
int cuGPUMemsetPrototype(size_t size, T *ptr, T value)
{
	T *dev_ptr;
	int error = checkCudaErrors(cuGetPointer((void**)&dev_ptr,ptr));

	int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / size;

    memset_kernel<T><<<numBlocks, blockSize>>>(size,dev_ptr,value);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

template< typename T >
int cuCopyGPUtoHostPrototype(int iw, int ih, float *source, T *dest)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	float *dev_source=source;
	T *dev_dest;
	int error = checkCudaErrors(cudaHostGetDevicePointer((void**)&dev_dest, (void*)dest, 0));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    copyGPUtoHost_kernel<T><<<blocks, threads>>>(iw, ih, is, dev_source, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

template< typename T >
int cuCopyHostToGPUPrototype(int iw, int ih, T *source, float *dest, float factor)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	T *dev_source;
	float *dev_dest=dest;

	int error = checkCudaErrors(cudaHostGetDevicePointer((void**)&dev_source, (void*)source, 0));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    copyHostToGPU_kernel<T><<<blocks, threads>>>(iw, ih, is, dev_source, dev_dest, factor);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

template< typename T >
int cuBoxFilterPrototype(int iw, int ih, T *source, T *dest, int bw, int bh)
{
    // allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
    T *dev_source, *dev_dest;
    int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
    error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    boxfilter_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_dest, bw, bh);
    error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

template< typename T >
int cuSobelFilterPrototype(int iw, int ih, T *source, T *dest)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
    T *dev_source, *dev_dest;
    int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
    error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    sobelfilter_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

template< typename T >
int cuImageDerivativePrototype(int iw, int ih, T *source, T *target, T *Ix, T *Iy, T *Iz, int type)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	T *dev_source, *dev_target, *dev_Ix, *dev_Iy, *dev_Iz;

	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
    error += checkCudaErrors(cuGetPointer((void**)&dev_target,target));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Ix,Ix));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Iy,Iy));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Iz,Iz));

	dim3 threads(16, 16);
	dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

	// Execute the kernel
	switch(type)
	{
		case 1:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 1, 0, 0, -1, 1, 0);
			break;
		case 2:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 1, 0, -1, 1, 0, 0);
			break;
		case 3:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 2, 0, -1, 0, 1, 0);
			break;
		case 4:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 2, 0, 0, -3, 4, -1);
			break;
		case 5:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 2, 1, -4, 3, 0, 0);
			break;
		default:
			derivative_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz, 12, 1, -8, 0, 8, -1);
			break;
	}
	error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

template< typename T >
int cuImageDerivativeExtendedPrototype(int iw, int ih, T *source, T *target,
									   T *Ix, T *Iy, T *Iz, T *Ixx, T *Iyy, T *Ixy, T *Ixz, T *Iyz)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	T *dev_source, *dev_target, *dev_Ix, *dev_Iy, *dev_Iz;
	T *dev_Ixx, *dev_Iyy, *dev_Ixy, *dev_Ixz, *dev_Iyz;

	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
    error += checkCudaErrors(cuGetPointer((void**)&dev_target,target));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Ix,Ix));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Iy,Iy));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Iz,Iz));
    error += checkCudaErrors(cuGetPointer((void**)&dev_Ixx,Ixx));
	error += checkCudaErrors(cuGetPointer((void**)&dev_Iyy,Iyy));
	error += checkCudaErrors(cuGetPointer((void**)&dev_Ixy,Ixy));
	error += checkCudaErrors(cuGetPointer((void**)&dev_Ixz,Ixz));
	error += checkCudaErrors(cuGetPointer((void**)&dev_Iyz,Iyz));

	dim3 threads(16, 16);
	dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

	// Execute the kernel
	derivative_kernel_extended<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_target, dev_Ix, dev_Iy, dev_Iz,
													  dev_Ixx, dev_Iyy, dev_Ixy, dev_Ixz, dev_Iyz,
													  12, 1, -8, 0, 8, -1,
													  12, -1, 16, -30, 16, -1);
	error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

template< typename T >
int cuWarpingPrototype(int iw, int ih, T *source, T *dest, float *dx, float *dy, bool isTotalDisp)
{
    // allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
    T *dev_source, *dev_dest;
    int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    // Execute the kernel
    warping_kernel<T><<<blocks,threads>>>(iw, ih, is, dev_source, dev_dest, dx, dy, isTotalDisp);
    error += checkCudaErrors(cudaThreadSynchronize());

    return error;
}

template< typename T >
int cuDownScalePrototype(int iw, int ih, T *source, T *dest)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw);
	int os = AlignBuffer(iw/2);
	T *dev_source, *dev_dest;
	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw, threads.x), iDivUp(ih, threads.y));

    downscale_kernel<T><<<blocks, threads>>>(iw, ih, is, os, dev_source, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

template< typename T >
int cuUpScalePrototype(int iw, int ih, float scale, T *source, T *dest)
{
	// allocate memory for the bitmap in GPU memory
	int is = AlignBuffer(iw*2);
	int os = AlignBuffer(iw);
	T *dev_source, *dev_dest;
	int error = checkCudaErrors(cuGetPointer((void**)&dev_source,source));
	error += checkCudaErrors(cuGetPointer((void**)&dev_dest,dest));

    dim3 threads(16, 16);
    dim3 blocks(iDivUp(iw*2, threads.x), iDivUp(ih*2, threads.y));

    upscale_kernel<T><<<blocks, threads>>>(iw*2, ih*2, is, os, scale, dev_source, dev_dest);
    error += checkCudaErrors(cudaThreadSynchronize());

	return error;
}

#endif /* CUDAIMAGEPROCHEADER_CUH_ */
