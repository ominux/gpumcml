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

/*  Multi-GPU support: 
Sets the maximum number of GPUs to 6 (assuming 3 dual-GPU cards - e.g., GTX 295) 
*/
#define MAX_GPU_COUNT 6

/*  Kernel execution configuration 
*/
#define NUM_BLOCKS 30
#define NUM_THREADS_PER_BLOCK 256  //512 for Fermi, 256 for GTX 280
#define NUM_THREADS (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)
#define WARP_SZ 32
#define NUM_WARPS_PER_BLK (NUM_THREADS_PER_BLOCK / WARP_SZ)

/*  Uncomment for old architecture (compute capability 1.1)
If enabled, 32-bit atomic instructions are used to 
perform/emulate 64-bit atomicAdd to global memory 
Check the NVIDIA programming guide to see what compute 
capability your graphics card supports
*/
// #define EMULATED_ATOMIC

/* Configure the L1 cache to have 16KB of shared memory and
48KB of hardware-managed cache.
If this flag is set, shared-memory-based caching is disabled.
*/
// #define USE_TRUE_CACHE

/* The MAX_IR x MAX_IZ portion of the absorption array is cached in
shared memory. We need to tune this to maximally utilized the resources.
*/
#define MAX_IR 24
#define MAX_IZ 64    //128 for Fermi, 64 for GTX 280

/* To reduce access conflicts to the absorption array <A_rz>, we allocate
multiple copies of <A_rz> in the global memory. We want to ensure that each
SM works on a separate copy of A_rz.
*/
//#define N_A_RZ_COPIES 30

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

