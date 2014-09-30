/*
 * gpuImageProcAPI.h
 *
 *  Created on: 7 Aug 2014
 *      Author: Mauricio Vanegas
 */

#ifndef GPUIMAGEPROCAPI_H_
#define GPUIMAGEPROCAPI_H_

#define FORWARD_FIRST_ORDER 	1
#define BACKWARD_FIRST_ORDER 	2
#define CENTERED_SECOND_ORDER 	3
#define FORWARD_SECOND_ORDER 	4
#define BACKWARD_SECOND_ORDER 	5
#define CENTERED_FOUR_ORDER 	0

int cuBoxFilter(int iw, int ih, unsigned char *source, unsigned char *dest, int bw, int bh);
int cuBoxFilter(int iw, int ih, float *source, float *dest, int bw, int bh);
int cuSobelFilter(int iw, int ih, unsigned char *source, unsigned char *dest);
int cuSobelFilter(int iw, int ih, float *source, float *dest);
int cuImageDerivative(int iw, int ih, unsigned char *source, unsigned char *target,
					unsigned char *Ix, unsigned char *Iy, unsigned char *Iz, int type);
int cuImageDerivative(int iw, int ih, float *source, float *target,
					float *Ix, float *Iy, float *Iz, int type);
int cuImageDerivativeExtended(int iw, int ih, unsigned char *source, unsigned char *target,
								unsigned char *Ix, unsigned char *Iy, unsigned char *Iz,
								unsigned char *Ixx, unsigned char *Iyy, unsigned char *Ixy,
								unsigned char *Ixz, unsigned char *Iyz);
int cuImageDerivativeExtended(int iw, int ih, float *source, float *target,
							  float *Ix, float *Iy, float *Iz, float *Ixx, float *Iyy, float *Ixy,
							  float *Ixz, float *Iyz);
int cuWarping(int iw, int ih, unsigned char *source, unsigned char *dest, float *dx, float *dy, bool isTotalDisp);
int cuWarping(int iw, int ih, float *source, float *dest, float *dx, float *dy, bool isTotalDisp);
int cuWarpingText(int iw, int ih, float *source, float *dest, float *dx, float *dy);
int cuDownScale(int iw, int ih, unsigned char *source, unsigned char *dest);
int cuDownScale(int iw, int ih, float *source, float *dest);
int cuUpScale(int iw, int ih, float scale, unsigned char *source, unsigned char *dest);
int cuUpScale(int iw, int ih, float scale, float *source, float *dest);
int cuUpScaleTexture(int iw, int ih, float scale, float *source, float *dest);
int cuDistortion(int iw, int ih, float *dx, float *dy);
int cuCartToPolar(int iw, int ih, float *dx, float *dy, int px, int py);
int cuPolarToCart(int iw, int ih, float *dx, float *dy, int px, int py);
int cuVectorAdd(float *vec1, float *vec2, int length, float *sum);
int cuJacobiSolver(float *u, float *v, float *du0, float *dv0, float *Ix, float *Iy, float *Iz,
                    int iw, int ih, float omeag, float alpha, float *du1, float *dv1);
int cuInfoGPUDevice(void);
int cuGPUInit(void);

int cuCopyGPUtoHost(int iw, int ih, float *source, unsigned char *dest);
int cuCopyGPUtoHost(int iw, int ih, float *source, float *dest);
int cuCopyHostToGPU(int iw, int ih, unsigned char *source, float *dest, float factor);
int cuCopyHostToGPU(int iw, int ih, float *source, float *dest, float factor);
int cuGPUMemset(unsigned char *ptr, unsigned char value, size_t size);
int cuGPUMemset(float *ptr, float value, size_t size);
int cuMemCopyGPUtoHost(void *dst, void *src, size_t bytes);
int cuMemCopyHostToGPU(void *dst, void *src, size_t bytes);
int	cuCreateSharedBuffer(size_t bytes, void **ptr);
int cuCreateGPUBuffer(size_t bytes, void **ptr);
int cuDestroySharedBuffer(void *ptr);
int cuDestroyGPUBuffer(void *ptr);

#endif /* GPUIMAGEPROCAPI_H_ */
