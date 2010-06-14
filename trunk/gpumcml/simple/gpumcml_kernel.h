/*****************************************************************************
 *
 *   Header file for GPU-related data structures and kernel configurations
 *
 ****************************************************************************/
/*	 
 *   This file is part of GPUMCML.
 * 
 *   GPUMCML is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   GPUMCML is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with GPUMCML.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _GPUMCML_KERNEL_H_
#define _GPUMCML_KERNEL_H_

#include "gpumcml.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*  Number of simulation steps performed by each thread in one kernel call
*/
#define NUM_STEPS 50000  //Use 5000 for faster response time

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Make sure __CUDA_ARCH__ is always defined by the user.
#ifdef _WIN32 
  #define __CUDA_ARCH__ 120
#endif

#ifndef __CUDA_ARCH__
#error "__CUDA_ARCH__ undefined!"
#endif

/**
 * Although this simple version of GPUMCML is not intended for high
 * performance, there are still a few parameters to be configured for
 * different GPUs.
 *
 * - NUM_BLOCKS:
 *      number of thread blocks in the grid to be launched
 *
 * - NUM_THREADS_PER_BLOCK:
 *      number of threads per thread block
 *
 * - EMULATED_ATOMIC:
 *      Enable this option for GPUs with Compute Capability 1.1,
 *      which do not support 64-bit atomicAdd to the global memory.
 *      In this case, we use two 32-bit atomicAdd's to emulate the
 *      64-bit version.
 *
 * - USE_TRUE_CACHE:
 *      Enable this option for GPUs with Compute Capability 2.0 (Fermi),
 *      which have a 64KB configurable L1 cache in each SM.
 *      If enabled, the L1 cache is configured to have 48KB of true cache
 *      and 16KB of shared memory, as opposed to 16KB of true cache and
 *      48KB of shared memory. Since the shared memory is not utilized
 *      in this simple version, you are encouraged to enable this option
 *      to cache more accesses to the absorption array in the global memory.
 */

/////////////////////////////////////////////
// Compute Capability 2.0
/////////////////////////////////////////////
#if __CUDA_ARCH__ == 200

#define NUM_BLOCKS 30
#define NUM_THREADS_PER_BLOCK 512
// #define EMULATED_ATOMIC
#define USE_TRUE_CACHE

/////////////////////////////////////////////
// Compute Capability 1.2 or 1.3
/////////////////////////////////////////////
#elif (__CUDA_ARCH__ == 120) || (__CUDA_ARCH__ == 130)

#define NUM_BLOCKS 30
#define NUM_THREADS_PER_BLOCK 256
#define EMULATED_ATOMIC

/////////////////////////////////////////////
// Compute Capability 1.1
/////////////////////////////////////////////
#elif (__CUDA_ARCH__ == 110)

#define NUM_BLOCKS 14       // should match the number of SMs on the GPUs
#define NUM_THREADS_PER_BLOCK 192
#define EMULATED_ATOMIC

/////////////////////////////////////////////
// Unsupported Compute Capability
/////////////////////////////////////////////
#else

#error "GPUMCML only supports compute capability 1.1 to 2.0!"

#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/**
 * Derived macros
 */

#define NUM_THREADS (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct __align__(16)
{
  FLOAT init_photon_w;      // initial photon weight 

  FLOAT dz;                 // z grid separation.[cm] 
  FLOAT dr;                 // r grid separation.[cm] 

  UINT32 na;                // array range 0..na-1. 
  UINT32 nz;                // array range 0..nz-1. 
  UINT32 nr;                // array range 0..nr-1. 

  UINT32 num_layers;        // number of layers. 
} SimParamGPU;

typedef struct __align__(16)
{
  FLOAT z0, z1;             // z coordinates of a layer. [cm] 
  FLOAT n;                  // refractive index of a layer. 

  FLOAT muas;               // mua + mus 
  FLOAT rmuas;              // 1/(mua+mus) 
  FLOAT mua_muas;           // mua/(mua+mus)

  FLOAT g;                  // anisotropy.

  FLOAT cos_crit0, cos_crit1;
} LayerStructGPU;

// The max number of layers supported (MAX_LAYERS including 2 ambient layers)
#define MAX_LAYERS 100

__constant__ SimParamGPU d_simparam;
__constant__ LayerStructGPU d_layerspecs[MAX_LAYERS];

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Thread-private states that live across batches of kernel invocations
// Each field is an array of length NUM_THREADS.
//
// We use a struct of arrays as opposed to an array of structs to enable
// global memory coalescing.
//
typedef struct
{
  // cartesian coordinates of the photon [cm]
  FLOAT *photon_x;
  FLOAT *photon_y;
  FLOAT *photon_z;

  // directional cosines of the photon
  FLOAT *photon_ux;
  FLOAT *photon_uy;
  FLOAT *photon_uz;

  FLOAT *photon_w;            // photon weight
  FLOAT *photon_sleft;        // leftover step size [cm]

  // index to layer where the photon resides
  UINT32 *photon_layer;

  UINT32 *is_active;          // is this thread active?
} GPUThreadStates;


typedef struct
{
  // cartesian coordinates of the photon [cm]
  FLOAT x;
  FLOAT y;
  FLOAT z;

  // directional cosines of the photon
  FLOAT ux;
  FLOAT uy;
  FLOAT uz;

  FLOAT w;            // photon weight

  FLOAT s;            // step size [cm]
  FLOAT sleft;        // leftover step size [cm]

  // index to layer where the photon resides
  UINT32 layer;

  // flag to indicate if photon hits a boundary
  UINT32 hit;
} PhotonStructGPU;

#endif // _GPUMCML_KERNEL_H_

