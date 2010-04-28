
/*****************************************************************************
 *
 * Defines GPU-related data structures, and kernel configurations.
 *
 ****************************************************************************/

#ifndef _CUDAMCML_KERNEL_H_
#define _CUDAMCML_KERNEL_H_

#include "cudamcml.h"

// Multi-GPU
#define MAX_GPU_COUNT 4

// kernel execution configuration
#define NUM_BLOCKS 30
#define NUM_THREADS_PER_BLOCK 256
#define NUM_THREADS (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)

// the number of simulation steps performed by each thread in one kernel
#define NUM_STEPS 50000

// Use the 32-bit atomicAdd to do 64-bit atomicAdd
// for devices with compute capability 1.1
// #define USE_ATOMICADD_ULL_CC1

/////////////////////////////
//Shared mem
/////////////////////////////

//NOTE: Make sure MAX_IR * MAX_IZ is less than 4096 (max - 16kB shared)
//and divisible by NUM_THREADS_PER_BLOCK for proper initialization to 0.
#define MAX_IR 28
#define MAX_IZ 128

//#define MAX_NINT 3584
//4096 - 512 reserved for kernel call and
//to allow NUM_THREADS_PER_BLOCK=512 to work in kernel
//4096 x 4 bytes = 16kB in shared mem

// the limit that indicates overflow of an element of A_rz in the shared mem
//
// This takes into account the worst case, i.e. all threads are adding to
// the same element in the current step.
// 
// Now this limit is generated dynamically (cudamcml_main.cu)
//
// #define MAX_OVERFLOW (4294967295 - WEIGHT_SCALE * NUM_THREADS_PER_BLOCK)

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct __align__(16)
{
    unsigned int num_layers;        /* number of layers. */
    unsigned int A_rz_overflow;     /* originally MAX_OVERFLOW */

    float init_photon_w;            /* initial photon weight */

    float dz;                       /* z grid separation.[cm] */
    float dr;                       /* r grid separation.[cm] */

    unsigned int na;                /* array range 0..na-1. */
    unsigned int nz;                /* array range 0..nz-1. */
    unsigned int nr;                /* array range 0..nr-1. */
} SimParamGPU;

typedef struct __align__(16)
{
    float z0, z1;       /* z coordinates of a layer. [cm] */
    float n;            /* refractive index of a layer. */

    float muas;         /* mua + mus */
    float rmuas;        /* 1/(mua+mus) */
    float mua_muas;     /* mua/(mua+mus) */

    float g;            /* anisotropy. */

    float cos_crit0, cos_crit1;
} LayerStructGPU;

// The max number of layers supported (MAX_LAYERS-2 usable layers)
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
    float *photon_x;
    float *photon_y;
    float *photon_z;

    // directional cosines of the photon
    float *photon_ux;
    float *photon_uy;
    float *photon_uz;

    float *photon_w;            // photon weight
    float *photon_sleft;        // leftover step size [cm]

    // index to layer where the photon resides
    unsigned int *photon_layer;

    unsigned int *is_active;    // is this thread active?
} GPUThreadStates;

#endif // _CUDAMCML_KERNEL_H_

