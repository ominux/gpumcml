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

/**
 * MCML kernel optimization parameters
 * You can tune them for the target GPU and the input model.
 *
 * - NUM_THREADS_PER_BLOCK:
 *      number of threads per thread block
 *
 * - USE_TRUE_CACHE:
 *      Use the true L1 cache instead of shared memory to cache
 *      updates to the absorption array A_rz. Configure the L1
 *      to have 48KB of true cache and 16KB of shared memory (unused).
 *      ** This feature is only available in Compute Capability 2.0.
 *
 * - MAX_IR, MAX_IZ:
 *      If shared memory is used to cache A_rz (i.e., USE_TRUE_CACHE
 *      is not set), cache the portion MAX_IR x MAX_IZ of A_rz.
 *
 * - USE_32B_ELEM_FOR_ARZ_SMEM:
 *      If shared memory is used to cache A_rz (i.e., USE_TRUE_CACHE
 *      is not set), each element of the MAX_IR x MAX_IZ portion can be
 *      either 32-bit or 64-bit. To use 32-bit, enable this option.
 *      Using 32-bit saves space and allows caching more of A_rz,
 *      but requires the explicit handling of element overflow.
 *
 * - N_A_RZ_COPIES:
 *      number of copies of A_rz allocated in global memory
 *      Each block is assigned a copy to write to in a round-robin fashion.
 *      Using more copies can reduce access contention, but it increases
 *      global memory usage and reduces the benefit of the L2 cache on
 *      Fermi GPUs (Compute Capability 2.0).
 *      This number should not exceed the number of thread blocks.
 *
 * - USE_64B_ATOMIC_SMEM:
 *      If the elements of A_rz cached in shared memory are 64-bit (i.e.
 *      USE_32B_ELEM_FOR_ARZ_SMEM is not set), atomically update data in the
 *      shared memory using 64-bit atomic instructions, as opposed to
 *      emulating it using two 32-bit atomic instructions.
 *      ** This feature is only available in Compute Capability 2.0.
 *
 * There are two potential parameters to tune:
 * - number of thread blocks
 * - the number of registers usaged by each thread
 *
 * For the first parameter, we think that it should be the same as the number
 * of SMs in a GPU, regardless of the GPU's Compute Capability. Therefore,
 * this is dynamically set in gpumcml_main.cu and not exposed as a tunable
 * parameter here.
 *
 * Since the second parameter is set at compile time, you have to tune it in
 * the makefile. This parameter is strongly correlated with parameter
 * NUM_THREADS_PER_BLOCK. Using more registers per thread forces
 * NUM_THREADS_PER_BLOCK to decrease (due to hardware resource constraint).
 */

// By default, we assume the lowest Compute Capability we support: 1.2.
#ifndef CUDA_ARCH
#define CUDA_ARCH 12
#endif

/////////////////////////////////////////////
// Compute Capability 2.0
/////////////////////////////////////////////
#if CUDA_ARCH == 20

#define NUM_THREADS_PER_BLOCK 896
// #define USE_TRUE_CACHE
#define MAX_IR 48
#define MAX_IZ 128
// #define USE_32B_ELEM_FOR_ARZ_SMEM
#define N_A_RZ_COPIES 4
// #define USE_64B_ATOMIC_SMEM

/////////////////////////////////////////////
// Compute Capability 1.2 or 1.3
/////////////////////////////////////////////
#elif CUDA_ARCH == 12 || CUDA_ARCH == 13

#define NUM_THREADS_PER_BLOCK 256
#define MAX_IR 28
#define MAX_IZ 128
#define USE_32B_ELEM_FOR_ARZ_SMEM
#define N_A_RZ_COPIES 1

/////////////////////////////////////////////
// Unsupported Compute Capability
/////////////////////////////////////////////
#else

#error "GPUMCML only supports compute capability 1.2 to 2.0!"

#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/**
 * Derived macros and typedefs
 *
 * You should not modify them unless you know what you are doing.
 */

#define WARP_SZ 32
#define NUM_WARPS_PER_BLK (NUM_THREADS_PER_BLOCK / WARP_SZ)

#ifdef USE_32B_ELEM_FOR_ARZ_SMEM
typedef UINT32 ARZ_SMEM_TY;
#else
typedef UINT64 ARZ_SMEM_TY;
#endif

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*  Number of simulation steps performed by each thread in one kernel call
*/
#define NUM_STEPS 50000  //Use 5000 for faster response time

/*  Multi-GPU support: 
    Sets the maximum number of GPUs to 6
    (assuming 3 dual-GPU cards - e.g., GTX 295) 
*/
#define MAX_GPU_COUNT 6

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct __align__(16)
{
  GFLOAT init_photon_w;      // initial photon weight 

  GFLOAT dz;                 // z grid separation.[cm] 
  GFLOAT dr;                 // r grid separation.[cm] 

  UINT32 na;                // array range 0..na-1. 
  UINT32 nz;                // array range 0..nz-1. 
  UINT32 nr;                // array range 0..nr-1. 

  UINT32 num_layers;        // number of layers. 
  UINT32 A_rz_overflow;     // overflow threshold for A_rz_shared
} SimParamGPU;

typedef struct __align__(16)
{
  GFLOAT z0, z1;             // z coordinates of a layer. [cm] 
  GFLOAT n;                  // refractive index of a layer. 

  GFLOAT muas;               // mua + mus 
  GFLOAT rmuas;              // 1/(mua+mus) 
  GFLOAT mua_muas;           // mua/(mua+mus)

  GFLOAT g;                  // anisotropy.

  GFLOAT cos_crit0, cos_crit1;
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
  GFLOAT *photon_x;
  GFLOAT *photon_y;
  GFLOAT *photon_z;

  // directional cosines of the photon
  GFLOAT *photon_ux;
  GFLOAT *photon_uy;
  GFLOAT *photon_uz;

  GFLOAT *photon_w;            // photon weight

  // index to layer where the photon resides
  UINT32 *photon_layer;

  UINT32 *is_active;          // is this thread active?
} GPUThreadStates;

typedef struct
{
  // cartesian coordinates of the photon [cm]
  GFLOAT x;
  GFLOAT y;
  GFLOAT z;

  // directional cosines of the photon
  GFLOAT ux;
  GFLOAT uy;
  GFLOAT uz;

  GFLOAT w;            // photon weight

  GFLOAT s;            // step size [cm]
  //GFLOAT sleft;        // leftover step size [cm]
  //removed as an optimization to reduce code divergence

  // index to layer where the photon resides
  UINT32 layer;

  // flag to indicate if photon hits a boundary
  UINT32 hit;
} PhotonStructGPU;

#endif // _GPUMCML_KERNEL_H_

