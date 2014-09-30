/*
 * PDESolver.h
 *
 *  Created on: Jun 3, 2014
 *      Author: Mauricio Vanegas
 */

#ifndef PDESOLVER_H_
#define PDESOLVER_H_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <assert.h>

#include "EventHandler.h"
#include "gpuImageProcAPI.h"

#define checkErrors(val)	check( (val), #val, __FILE__, __LINE__ )
#define AlignBuffer(val)	iAlignUp( val, 32 )

/* Error handling */
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if(result)
	{
		fprintf(stderr,"PDESolver error calling \"%s\" at %s:%d. "
				"Please take a look at the above errors.\n", func, file, line);
		exit(EXIT_FAILURE);
	}
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

// swap two values
template<typename T>
inline void ptrSwap(T &a, T &b)
{
	T t = a;
	a = b;
	b = t;
}

class PDESolver : private EventHandler
{
public:
	PDESolver();
	~PDESolver();
	static PDESolver* getInstance()
	{
		if(!k_Instance)
		{
			k_Instance = new PDESolver();
			assert(k_Instance != NULL);
		}
		return k_Instance;
	}
	/* 					Function: getGPUDeviceInfo						*
	 * This function prints in the terminal the most relevant features	*
	 * of the GPU available in the system.							 	*/
	void getGPUDeviceInfo(void);

	/* 					Function: callbackRegistration					*
	 * CALLBACKREGISTRATION registers a function which shall be invoked	*
	 * each BASETIME milliseconds. By default BASETIME is set to 10. 	*/
	void callbackRegistration(void (*function)());
	void callbackRegistration(void (*function)(), int baseTime);

	/* 					Function: populateGPUBuffer						*
	 * This function copies data from a GPU/Host unified memory space 	*
	 * to a GPU memory space. This function can be used also for		*
	 * normalising the input data by means of the FACTOR parameter.		*/
	void populateGPUBuffer(float *dst, unsigned char *src, int width, int height, float factor);
	void populateGPUBuffer(float *dst, float *src, int width, int height, float factor);
	void populateGPUBuffer(float *dst, unsigned char *src, int width, int height);
	void populateGPUBuffer(float *dst, float *src, int width, int height);

	/* 					Function: retrieveGPUBuffer						*
	 * This function copies data from a GPU memory space to a GPU/Host 	*
	 * unified memory space.											*/
	void retrieveGPUBuffer(unsigned char *dst, float *src, int width, int height);
	void retrieveGPUBuffer(float *dst, float *src, int width, int height);

	/* 					Function: allocateSharedMemory					*
	 * This function allocates CPU memory which is automatically mapped	*
	 * into the GPU memory space. 										*/
	template<typename T>
	T* allocateSharedMemory(size_t width, size_t height)
	{
		T* ptr = NULL;
		size_t bytes = width*height;
		checkErrors(cuCreateSharedBuffer(bytes*sizeof(T), (void**)&ptr));

		this->alloc_host_memory.push_back(ptr);
		return ptr;
	}

	/* 					Function: deallocateSharedMemory				*
	 * This function deallocate memory allocated by means of the method	*
	 * allocateSharedMemory.											*/
	void deallocateSharedMemory(void);

	/* 					Function: allocateCudaMemory					*
	 * This function allocates GPU memory space. The allocated memory is*
	 * conceived for coalescing. In order to guarantee a proper memory 	*
	 * access for all the class methods, it must be used the method 	*
	 * populateGPUBuffer for populating the allocated memory. To		*
	 * adequately retrieve the data in the allocated memory, it must be	*
	 * used the method retrieveGPUBuffer.								*/
	float* allocateCudaMemory(size_t width, size_t height);

	/* 					Function: deallocateCudaMemory					*
	 * This function deallocate memory allocated by means of the method	*
	 * allocateCudaMemory.												*/
	void deallocateCudaMemory(void);

	/* 						Function: setValueToGPUMem					*
	 * This function copies VALUE to the memory space which is SIZE		*
	 * long. PTR can point to both device or GPU/Host unified memory	*
	 * space.															*/
	void setValueToGPUMem(unsigned char *ptr, unsigned char value, size_t size);
	void setValueToGPUMem(float *ptr, float value, size_t size);

	/* 						Function: memcopyGPUtoHost					*
	 * This function uses cudaMemcpy for writing from GPU memory space	*
	 * to GPU/Host unified memory space.								*/
	void memcopyGPUtoHost(void *dst, void *src, size_t bytes);

	/* 						Function: memcopyHostToGPU					*
	 * This function uses cudaMemcpy for writing from GPU/Host unified 	*
	 * memory space	to GPU memory space.								*/
	void memcopyHostToGPU(void *dst, void *src, size_t bytes);

	/* 					Function: spatialImDerivative					*
	 * This function performs spatial derivatives with different type	*
	 * of approximations (forward, backward, centered) and order  		*
	 * approximations (first, second, and fourth order). The argument	*
	 * TYPE is used for selecting the numerical approximation. The 		*
	 * overloaded functions with second derivatives use centered 		*
	 * difference approximation of order fourth for spatial derivatives.*
	 * Finally, the temporal derivative IZ is obtained by subtracting	*
	 * TARIM from SRCIM.									 			*/
	void spatialImDerivative(unsigned char *srcIm, unsigned char *tarIm,
							unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
							int width, int height, int type);
	void spatialImDerivative(float *srcIm, float *tarIm, float *Ix, float *Iy, float *Iz,
							int width, int height, int type);
	void spatialImDerivative(unsigned char *srcIm, unsigned char *tarIm,
							unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
							unsigned char *Ixx, unsigned char *Iyy, unsigned char *Ixy,
							unsigned char *Ixz, unsigned char *Iyz, int width, int height);
	void spatialImDerivative(float *srcIm, float *tarIm, float *Ix, float *Iy, float *Iz,
							 float *Ixx, float *Iyy, float *Ixy, float *Ixz, float *Iyz,
							 int width, int height);

	/* 						Function: warping							*
	 * This function performs the image warping according to the matrix	*
	 * displacements DX and DY. This method can take DX and DY as both	*
	 * a total or partial pixel displacement by using the parameter		*
	 * "isTotalDisp". By default "isTotalDisp" is set to false.			*/
	void warping(unsigned char *srcIm, unsigned char *dstIm, float *dx, float *dy,
					int width, int height);
	void warping(unsigned char *srcIm, unsigned char *dstIm, float *dx, float *dy,
					int width, int height, bool isTotalDisp);
	void warping(float *srcIm, float *dstIm, float *dx, float *dy,
					int width, int height);
	void warping(float *srcIm, float *dstIm, float *dx, float *dy,
					int width, int height, bool isTotalDisp);
	void warpingText(float *srcIm, float *dstIm, float *dx, float *dy,
					int width, int height);

	/* 						Function: downSampling						*
	 * This function performs a down sampling of the image by a scale	*
	 * of two. Parameters WIDTH and HEIGHT are related to the source	*
	 * image.															*/
	void downSampling(unsigned char *srcIm, unsigned char *dstIm, int width, int height);
	void downSampling(float *srcIm, float *dstIm, int width, int height);

	/* 						Function: upSampling						*
	 * This function performs an up sampling of the image by a scale	*
	 * of two. Parameters WIDTH and HEIGHT are related to the source	*
	 * image.															*/
	void upSampling(unsigned char *srcIm, unsigned char *dstIm, float scale,
					int width, int height);
	void upSampling(float *srcIm, float *dstIm, float scale, int width, int height);
	void upSamplingTexture(float *srcIm, float *dstIm, float scale, int width, int height);

	/* 						Function: sobelFunc							*
	 * This function performs Sobel filter. 							*/
	void sobelFunc(unsigned char *srcIm, unsigned char *dstIm, int width, int height);
	void sobelFunc(float *srcIm, float *dstIm, int width, int height);

	/* 						Function: blurredFilter						*
	 * This function performs a blurred process on the image. 			*/
	void blurredFilter(unsigned char *srcIm, unsigned char *dstIm, int width, int height);
	void blurredFilter(float *srcIm, float *dstIm, int width, int height);

	/* 							Function: destroy						*
	 * This function must be used at the end of your code in order to	*
	 * deallocate the class singleton. It also deallocate all the 		*
	 * memory allocated by the methods provided.						*/
	void destroy(void);

	/* 						Function: distortionLutGen					*
	 * This function generates luts for displacements in image 			*
	 * coordinates X and Y.	DX and DY must be allocated as GPU memory.	*
	 * It can be used the method allocateCudaMemory for allocating DX	*
	 * and DY.															*/
	void distortionLutGen(float *dx, float *dy, int width, int height);

	/* 						Function: cartToPolarLutGen					*
	 * This function generates luts for displacements in image 			*
	 * coordinates X and Y. These displacements perform a transformation*
	 * in the image coordinate system (Cartesian to Polar).	DX and DY	*
	 * must be allocated as GPU memory.	It can be used the method 		*
	 * allocateCudaMemory for allocating DX	and DY.						*/
	void cartToPolarLutGen(float *dx, float *dy, int width, int height, int px, int py);

	/* 						Function: polarToCartLutGen					*
	 * This function generates luts for displacements in image 			*
	 * coordinates X and Y. These displacements perform a transformation*
	 * in the image coordinate system (Polar to Cartesian).	DX and DY	*
	 * must be allocated as GPU memory.	It can be used the method 		*
	 * allocateCudaMemory for allocating DX	and DY.						*/
	void polarToCartLutGen(float *dx, float *dy, int width, int height, int px, int py);

	/* 						Function: memcopyHostToGPU					*
	 * This function performs one iteration of Jacobi method for a		*
	 * corresponding linear system.	This method only uses GPU allocated	*
	 * memory.															*/
	void jacobiIteration(float *u, float *v, float *du0, float *dv0, float *Ix, float *Iy, float *Iz,
            			int w, int h, float omega, float alpha, float *du1, float *dv1);

	/* 							Function: vectorAdd						*
	 * This function performs a vector sum.								*/
	void vectorAdd(float *vec1, float *vec2, int length, float *sum);

private:
	/* PDESolver object... */
	static PDESolver* k_Instance;

	/* Variables */
	std::vector<void *> alloc_host_memory;
	std::vector<void *> alloc_cuda_memory;
};

#endif /* PDESOLVER_H_ */
