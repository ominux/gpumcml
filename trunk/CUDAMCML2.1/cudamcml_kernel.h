/*****************************************************************************
 *
 * Defines GPU-related data structures, and kernel configurations.
 *
 ****************************************************************************/

#ifndef _CUDAMCML_KERNEL_H_
#define _CUDAMCML_KERNEL_H_

#include "cudamcml.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// the number of simulation steps performed by each thread in one kernel
//
#define NUM_STEPS 50000 //Use 5000 for faster response time

// Multi-GPU support
// Sets the maximum number of GPUs to 6 (assuming 3 dual-GPU cards - e.g., GTX 295)
//
#define MAX_GPU_COUNT 6

// kernel execution configuration
//
#define NUM_BLOCKS 30
#define NUM_THREADS_PER_BLOCK 256  //512 for Fermi, 256 for GTX 280
#define NUM_THREADS (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)
#define WARP_SZ 32
#define NUM_WARPS_PER_BLK (NUM_THREADS_PER_BLOCK / WARP_SZ)

// Uncomment for FERMI architecture 
// If disabled, 32-bit atomic instructions are used to 
// perform 64-bit atomicAdd to shared memory 
// (used by pre-FERMI generation, e.g., GTX 280)

// #define FERMI


// Configure the L1 cache to have 16KB of shared memory and
// 48KB of hardware-managed cache.
// If this flag is set, shared-memory-based caching is disabled.
//
// #define USE_TRUE_CACHE

// The MAX_IR x MAX_IZ portion of the absorption array is cached in
// shared memory. We need to tune this to maximally utilized the resources.
//

#define MAX_IR 24
#define MAX_IZ 64    //128 for Fermi, 64 for GTX 280

// To reduce access conflicts to the absorption array <A_rz>, we allocate
// multiple copies of <A_rz> in the global memory. We want to ensure that each
// SM works on a separate copy of A_rz.
//
//#define N_A_RZ_COPIES 30

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct __align__(16)
{
    FLOAT init_photon_w;            /* initial photon weight */

    FLOAT dz;                       /* z grid separation.[cm] */
    FLOAT dr;                       /* r grid separation.[cm] */

    UINT32 na;                /* array range 0..na-1. */
    UINT32 nz;                /* array range 0..nz-1. */
    UINT32 nr;                /* array range 0..nr-1. */

    UINT32 num_layers;        /* number of layers. */
} SimParamGPU;

typedef struct __align__(16)
{
    FLOAT z0, z1;       /* z coordinates of a layer. [cm] */
    FLOAT n;            /* refractive index of a layer. */

    FLOAT muas;         /* mua + mus */
    FLOAT rmuas;        /* 1/(mua+mus) */
    FLOAT mua_muas;     /* mua/(mua+mus) */

    FLOAT g;            /* anisotropy. */

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

#endif // _CUDAMCML_KERNEL_H_

