/*****************************************************************************
*
*   Kernel code for GPUMCML
*   =========================================================================
*   Featured Optimizations: 
*   1) Shared memory cache for high fluence region  
*   2) Reduced divergence 
*   3) Optimized atomicAdd
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

#ifndef _GPUMCML_KERNEL_CU_
#define _GPUMCML_KERNEL_CU_

#include "gpumcml_kernel.h"
#include "gpumcml_rng.cu"

//////////////////////////////////////////////////////////////////////////////
// This host routine computes the maximum element value of A_rz in shared
// memory, that indicates an imminent overflow.
//
// This MAX_OVERFLOW is MAX_UINT32 - MAX(dwa) * NUM_THREADS_PER_BLOCK.
//
// All we really need to compute is
//    MAX(dwa) <= WEIGHT_SCALE * <init_photon_w> * MAX( mua/(mua+mus) )
//
// We have to be accurate in this bound because if we assume that
//    MAX(dwa) = WEIGHT_SCALE,
// MAX_OVERFLOW can be small if WEIGHT_SCALE is large, like 12000000.
//
// <n_layers> is the length of <layers>, excluding the top and bottom layers.
//////////////////////////////////////////////////////////////////////////////
UINT32 compute_Arz_overflow_count(FLOAT init_photon_w,
        LayerStruct *layers, UINT32 n_layers, UINT32 n_threads_per_tblk)
{
    // Determine the largest mua/(mua+mus) over all layers.
    double max_muas = 0;
    for (int i = 1; i <= n_layers; ++i)
    {
        double muas = layers[i].mua * layers[i].mutr;
        if (max_muas < muas) max_muas = muas;
    }

    // Determine an upper bound of <dwa> in <MCMLKernel>.
    UINT32 max_dwa = (UINT32)(init_photon_w * max_muas * WEIGHT_SCALE) + 1;

    return (0xFFFFFFFF - max_dwa * n_threads_per_tblk);
}

//////////////////////////////////////////////////////////////////////////////
//   Initialize photon position (x, y, z), direction (ux, uy, uz), weight (w), 
//   and current layer (layer).
//   Note: Infinitely narrow beam (pointing in the +z direction = downwards)
//////////////////////////////////////////////////////////////////////////////
__device__ void LaunchPhoton(FLOAT *x, FLOAT *y, FLOAT *z,
                             FLOAT *ux, FLOAT *uy, FLOAT *uz,
                             FLOAT *w, UINT32 *layer)
{
  *x = *y = *z = MCML_FP_ZERO;
  *ux = *uy = MCML_FP_ZERO;
  *uz = FP_ONE;
  *w = d_simparam.init_photon_w;
  *layer = 1;
}

//////////////////////////////////////////////////////////////////////////////
//   Compute the step size for a photon packet when it is in tissue
//   Calculate new step size: -log(rnd)/(mua+mus).
//////////////////////////////////////////////////////////////////////////////
__device__ void ComputeStepSize(UINT32 layer,
                                FLOAT *s_ptr,
                                UINT64 *rnd_x, UINT32 *rnd_a)
{
	*s_ptr = -logf(rand_MWC_oc(rnd_x,rnd_a))* d_layerspecs[layer].rmuas;

}

//////////////////////////////////////////////////////////////////////////////
//   Check if the step size calculated above will cause the photon to hit the 
//   boundary between 2 layers.
//   Return 1 for a hit, 0 otherwise.
//   If the projected step hits the boundary, the photon steps to the boundary
//////////////////////////////////////////////////////////////////////////////
__device__ int HitBoundary(UINT32 layer, FLOAT z, FLOAT uz,
                           FLOAT *s_ptr)
{
  /* step size to boundary. */
  FLOAT dl_b; 

  /* Distance to the boundary. */
  FLOAT z_bound = (uz > MCML_FP_ZERO) ?
    d_layerspecs[layer].z1 : d_layerspecs[layer].z0;
  dl_b = __fdividef(z_bound - z, uz);     // dl_b > 0

  FLOAT s = *s_ptr;
  UINT32 hit_boundary = (uz != MCML_FP_ZERO) && (s > dl_b);
  if (hit_boundary)
  {
    // No need to multiply by (mua + mus), as it is later
    // divided by (mua + mus) anyways (in the original version).
    *s_ptr = dl_b;
  }

  return hit_boundary;
}

//////////////////////////////////////////////////////////////////////////////
//   Move the photon by step size (s) along direction (ux,uy,uz) 
//////////////////////////////////////////////////////////////////////////////
__device__ void Hop(FLOAT s, FLOAT ux, FLOAT uy, FLOAT uz,
                    FLOAT *x, FLOAT *y, FLOAT *z)
{
  *x += s * ux;
  *y += s * uy;
  *z += s * uz;
}

//////////////////////////////////////////////////////////////////////////////
//   UltraFast version (featuring reduced divergence compared to CPU-MCML)
//   If a photon hits a boundary, determine whether the photon is transmitted
//   into the next layer or reflected back by computing the internal reflectance
//////////////////////////////////////////////////////////////////////////////
__device__ void FastReflectTransmit(FLOAT x, FLOAT y, SimState *d_state_ptr,
                                    FLOAT *ux, FLOAT *uy, FLOAT *uz,
                                    UINT32 *layer, FLOAT* w,
                                    UINT64 *rnd_x, UINT32 *rnd_a)
{
  /* Collect all info that depend on the sign of "uz". */
  FLOAT cos_crit;
  UINT32 new_layer;
  if (*uz > MCML_FP_ZERO)
  {
    cos_crit = d_layerspecs[(*layer)].cos_crit1;
    new_layer = (*layer)+1;
  }
  else
  {
    cos_crit = d_layerspecs[(*layer)].cos_crit0;
    new_layer = (*layer)-1;
  }

  // cosine of the incident angle (0 to 90 deg)
  FLOAT ca1 = fabsf(*uz);

  // The default move is to reflect.
  *uz = -(*uz);

  // Moving this check down to "RFresnel = MCML_FP_ZERO" slows down the
  // application, possibly because every thread is forced to do
  // too much.
  if (ca1 > cos_crit)
  {
    /* Compute the Fresnel reflectance. */

    // incident and transmit refractive index
    FLOAT ni = d_layerspecs[(*layer)].n;
    FLOAT nt = d_layerspecs[new_layer].n;
    FLOAT ni_nt = __fdividef(ni, nt);   // reused later

    FLOAT sa1 = sqrtf(FP_ONE-ca1*ca1);
    FLOAT sa2 = fminf(ni_nt * sa1, FP_ONE);
    if (ca1 > COSZERO) sa2 = sa1;
    FLOAT uz1 = sqrtf(FP_ONE-sa2*sa2);    // uz1 = ca2

    FLOAT ca1ca2 = ca1 * uz1;
    FLOAT sa1sa2 = sa1 * sa2;
    FLOAT sa1ca2 = sa1 * uz1;
    FLOAT ca1sa2 = ca1 * sa2;

    FLOAT cam = ca1ca2 + sa1sa2; /* c- = cc + ss. */
    FLOAT sap = sa1ca2 + ca1sa2; /* s+ = sc + cs. */
    FLOAT sam = sa1ca2 - ca1sa2; /* s- = sc - cs. */

    FLOAT rFresnel = __fdividef(sam, sap*cam);
    rFresnel *= rFresnel;
    rFresnel *= (ca1ca2*ca1ca2 + sa1sa2*sa1sa2);

    // Hope "uz1" is very close to "ca1".
    if (ca1 > COSZERO) rFresnel = MCML_FP_ZERO;
    // In this case, we do not care if "uz1" is exactly 0.
    if (ca1 < COSNINETYDEG || sa2 == FP_ONE) rFresnel = FP_ONE;

    FLOAT rand = rand_MWC_co(rnd_x, rnd_a);

    if (rFresnel < rand)
    {
      // The move is to transmit.
      *layer = new_layer;

      // Let's do these even if the photon is dead.
      *ux *= ni_nt;
      *uy *= ni_nt;

      *uz = -copysignf(uz1, *uz);

      if (*layer == 0 || *layer > d_simparam.num_layers)
      {
        // transmitted
        FLOAT uz2 = *uz;
        UINT64 *ra_arr = d_state_ptr->Tt_ra;
        if (*layer == 0)
        {
          // diffuse reflectance
          uz2 = -uz2;
          ra_arr = d_state_ptr->Rd_ra;
        }

        UINT32 ia = acosf(uz2) * FP_TWO * RPI * d_simparam.na;
        UINT32 ir = __fdividef(sqrtf(x*x+y*y), d_simparam.dr);
        if (ir >= d_simparam.nr) ir = d_simparam.nr - 1;

        atomicAdd(&ra_arr[ia * d_simparam.nr + ir],
          (UINT32)(*w * WEIGHT_SCALE));

        // Kill the photon.
        *w = MCML_FP_ZERO;
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////
//   Computing the scattering angle and new direction by 
//	 sampling the polar deflection angle theta and the
// 	 azimuthal angle psi.
//////////////////////////////////////////////////////////////////////////////
__device__ void Spin(FLOAT g, FLOAT *ux, FLOAT *uy, FLOAT *uz,
                     UINT64 *rnd_x, UINT32 *rnd_a)
{
  FLOAT cost, sint; // cosine and sine of the polar deflection angle theta
  FLOAT cosp, sinp; // cosine and sine of the azimuthal angle psi
  FLOAT psi;
  FLOAT SIGN;
  FLOAT temp;
  FLOAT last_ux, last_uy, last_uz;
  FLOAT rand;

  /***********************************************************
  *	>>>>>>> SpinTheta
  *  Choose (sample) a new theta angle for photon propagation
  *	according to the anisotropy.
  *
  *	If anisotropy g is 0, then
  *		cos(theta) = 2*rand-1.
  *	otherwise
  *		sample according to the Henyey-Greenstein function.
  *
  *	Returns the cosine of the polar deflection angle theta.
  ****/

  rand = rand_MWC_co(rnd_x, rnd_a);

  cost = FP_TWO * rand - FP_ONE;

  if (g != MCML_FP_ZERO)
  {
    temp = __fdividef((FP_ONE - g * g), FP_ONE + g*cost);
    cost = __fdividef(FP_ONE + g * g - temp*temp, FP_TWO * g);
    cost = fmaxf(cost, -FP_ONE);
    cost = fminf(cost, FP_ONE);
  }
  sint = sqrtf(FP_ONE - cost * cost);

  /* spin psi 0-2pi. */
  rand = rand_MWC_co(rnd_x, rnd_a);

  psi = FP_TWO * PI_const * rand;
  __sincosf(psi, &sinp, &cosp);

  FLOAT stcp = sint * cosp;
  FLOAT stsp = sint * sinp;

  last_ux = *ux;
  last_uy = *uy;
  last_uz = *uz;

  if (fabsf(last_uz) > COSZERO) 
    /* normal incident. */
  {
    *ux = stcp;
    *uy = stsp;
    SIGN = ((last_uz) >= MCML_FP_ZERO ? FP_ONE : -FP_ONE);
    *uz = cost * SIGN;
  }
  else 
    /* regular incident. */
  {
    temp = rsqrtf(FP_ONE - last_uz * last_uz);
    *ux = (stcp * last_ux * last_uz - stsp * last_uy) * temp
      + last_ux * cost;
    *uy = (stcp * last_uy * last_uz + stsp * last_ux) * temp
      + last_uy * cost;
    *uz = __fdividef(-stcp, temp) + last_uz * cost;
  }
}

//////////////////////////////////////////////////////////////////////////////
//   Initialize thread states (tstates), created to allow a large 
//   simulation to be broken up into batches 
//   (avoiding display driver time-out errors)
//////////////////////////////////////////////////////////////////////////////
__global__ void InitThreadState(GPUThreadStates tstates)
{
  FLOAT photon_x, photon_y, photon_z;
  FLOAT photon_ux, photon_uy, photon_uz;
  FLOAT photon_w;
  UINT32 photon_layer;

  // Initialize the photon and copy into photon_<parameter x>
  LaunchPhoton(&photon_x, &photon_y, &photon_z,
    &photon_ux, &photon_uy, &photon_uz,
    &photon_w, &photon_layer);

  // This is the unique ID for each thread (or thread ID = tid)
  UINT32 tid = blockIdx.x * blockDim.x + threadIdx.x;

  tstates.photon_x[tid] = photon_x;
  tstates.photon_y[tid] = photon_y;
  tstates.photon_z[tid] = photon_z;
  tstates.photon_ux[tid] = photon_ux;
  tstates.photon_uy[tid] = photon_uy;
  tstates.photon_uz[tid] = photon_uz;
  tstates.photon_w[tid] = photon_w;
  tstates.photon_layer[tid] = photon_layer;

  tstates.is_active[tid] = 1;
}

//////////////////////////////////////////////////////////////////////////////
//   Save thread states (tstates), by copying the current photon 
//   data from registers into global memory
//////////////////////////////////////////////////////////////////////////////
__device__ void SaveThreadState(SimState *d_state, GPUThreadStates *tstates,
                                FLOAT photon_x, FLOAT photon_y, FLOAT photon_z,
                                FLOAT photon_ux, FLOAT photon_uy, FLOAT photon_uz,
                                FLOAT photon_w, UINT32 photon_layer,
                                UINT64 rnd_x, UINT32 rnd_a,
                                UINT32 is_active)
{
  UINT32 tid = blockIdx.x * blockDim.x + threadIdx.x;

  d_state->x[tid] = rnd_x;
  d_state->a[tid] = rnd_a;

  tstates->photon_x[tid] = photon_x;
  tstates->photon_y[tid] = photon_y;
  tstates->photon_z[tid] = photon_z;
  tstates->photon_ux[tid] = photon_ux;
  tstates->photon_uy[tid] = photon_uy;
  tstates->photon_uz[tid] = photon_uz;
  tstates->photon_w[tid] = photon_w;
  tstates->photon_layer[tid] = photon_layer;

  tstates->is_active[tid] = is_active;
}

//////////////////////////////////////////////////////////////////////////////
//   Restore thread states (tstates), by copying the latest photon 
//   data from global memory back into the registers
//////////////////////////////////////////////////////////////////////////////
__device__ void RestoreThreadState(SimState *d_state, GPUThreadStates *tstates,
                                   FLOAT *photon_x, FLOAT *photon_y, FLOAT *photon_z,
                                   FLOAT *photon_ux, FLOAT *photon_uy, FLOAT *photon_uz,
                                   FLOAT *photon_w, UINT32 *photon_layer,
                                   UINT64 *rnd_x, UINT32 *rnd_a,
                                   UINT32 *is_active)
{
  UINT32 tid = blockIdx.x * blockDim.x + threadIdx.x;

  *rnd_x = d_state->x[tid];
  *rnd_a = d_state->a[tid];

  *photon_x = tstates->photon_x[tid];
  *photon_y = tstates->photon_y[tid];
  *photon_z = tstates->photon_z[tid];
  *photon_ux = tstates->photon_ux[tid];
  *photon_uy = tstates->photon_uy[tid];
  *photon_uz = tstates->photon_uz[tid];
  *photon_w = tstates->photon_w[tid];
  *photon_layer = tstates->photon_layer[tid];

  *is_active = tstates->is_active[tid];
}

//////////////////////////////////////////////////////////////////////////////
// Flush the element at offset <s_addr> of A_rz in shared memory (s_A_rz)
// to the global memory (g_A_rz). <s_A_rz> is of dimension MAX_IR x MAX_IZ.
//////////////////////////////////////////////////////////////////////////////
__device__ void Flush_Arz(UINT64 *g_A_rz, ARZ_SMEM_TY *s_A_rz, UINT32 s_addr)
{
  UINT32 ir = s_addr / MAX_IZ;
  UINT32 iz = s_addr - ir * MAX_IZ;
  UINT32 g_addr = ir * d_simparam.nz + iz;

  atomicAdd(&g_A_rz[g_addr], (UINT64)s_A_rz[s_addr]);
}

//////////////////////////////////////////////////////////////////////////////
//   AtomicAdd to Shared Mem for Unsigned Long Long (ULL) data type
//   Note: Only Fermi architecture supports 64-bit atomicAdd to 
//   both shared memory and global memory 
//////////////////////////////////////////////////////////////////////////////
__device__ void AtomicAddULL_Shared(UINT64* address, UINT32 add)
{
#ifdef USE_64B_ATOMIC_SMEM
  atomicAdd(address, (UINT64)add);
#else
  if (atomicAdd((UINT32*)address,add) +add < add)
  {
    atomicAdd(((UINT32*)address)+1, 1U);
  }
#endif
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern __shared__ UINT32 MCMLKernel_smem[];

//////////////////////////////////////////////////////////////////////////////
//   Main Kernel for MCML (Calls the above inline device functions)
//////////////////////////////////////////////////////////////////////////////

template <int ignoreAdetection>
__global__ void MCMLKernel(SimState d_state, GPUThreadStates tstates)
{
  // photon structure stored in registers
  FLOAT photon_x, photon_y ,photon_z;
  FLOAT photon_ux, photon_uy, photon_uz;
  FLOAT photon_w;
  UINT32 photon_layer;

  // random number seeds
  UINT64 rnd_x;
  UINT32 rnd_a;

  // Flag to indicate if this thread is active
  UINT32 is_active;

  // Restore the thread state from global memory.
  RestoreThreadState(&d_state, &tstates,
    &photon_x, &photon_y, &photon_z,
    &photon_ux, &photon_uy, &photon_uz,
    &photon_w, &photon_layer,
    &rnd_x, &rnd_a,
    &is_active);

  //////////////////////////////////////////////////////////////////////////

  // Coalesce consecutive weight drops to the same address.
  UINT32 last_w = 0;
  UINT32 last_ir = 0, last_iz = 0, last_addr = 0;

  //////////////////////////////////////////////////////////////////////////

#ifndef USE_TRUE_CACHE
  // Cache the frequently acessed region of A_rz in the shared memory.
  __shared__ ARZ_SMEM_TY A_rz_shared[MAX_IR*MAX_IZ];

  if (ignoreAdetection == 0)
  {
    // Clear the cache.
    for (int i = threadIdx.x; i < MAX_IR*MAX_IZ; i += blockDim.x)
    {
      A_rz_shared[i] = 0;
    }
    __syncthreads();
  }

#ifdef USE_32B_ELEM_FOR_ARZ_SMEM
  // Overflow handling:
  //
  // It is too spacious to keep track of whether or not each element in
  // the shared memory is about to overflow. Therefore, we divide all the
  // elements into NUM_THREADS_PER_BLOCK groups (cyclic distribution). For
  // each group, we use a single flag to keep track of if ANY element in it
  // is about to overflow. This results in the following array.
  //
  // At the end of each simulation step, if the flag for any of the groups
  // is set, the corresponding thread (with id equal to the group index)
  // flushes ALL elements in the group to the global memory.
  //
  // This array is dynamically allocated.
  //
  UINT32 *A_rz_overflow = (UINT32*)MCMLKernel_smem;
  if (ignoreAdetection == 0)
  {
    // Clear the flags.
    A_rz_overflow[threadIdx.x] = 0;
  }
#endif

#endif

  //////////////////////////////////////////////////////////////////////////

  // Get the copy of A_rz (in the global memory) this thread writes to.
  UINT64 *g_A_rz = d_state.A_rz
    + (blockIdx.x % N_A_RZ_COPIES) * (d_simparam.nz * d_simparam.nr);

  //////////////////////////////////////////////////////////////////////////

  for (int iIndex = 0; iIndex < NUM_STEPS; ++iIndex)
  {
    // Only process photon if the thread is active.
    if (is_active)
    {
      FLOAT photon_s;     // current step size

      //>>>>>>>>> StepSizeInTissue() in MCML
      ComputeStepSize(photon_layer, &photon_s,
        &rnd_x, &rnd_a);

      //>>>>>>>>> HitBoundary() in MCML
      UINT32 photon_hit = HitBoundary(photon_layer,
        photon_z, photon_uz, &photon_s);

      Hop(photon_s, photon_ux, photon_uy, photon_uz,
        &photon_x, &photon_y, &photon_z);

      if (photon_hit)
      {
        FastReflectTransmit(photon_x, photon_y, &d_state,
          &photon_ux, &photon_uy, &photon_uz,
          &photon_layer, &photon_w,
          &rnd_x, &rnd_a);
      }
      else
      {
        //>>>>>>>>> Drop() in MCML
        FLOAT dwa = photon_w * d_layerspecs[photon_layer].mua_muas;
        photon_w -= dwa;

        if (ignoreAdetection == 0)
        {
          // automatic __float2uint_rz
          UINT32 iz = __fdividef(photon_z, d_simparam.dz);
          // automatic __float2uint_rz
          UINT32 ir = __fdividef(
            sqrtf(photon_x * photon_x + photon_y * photon_y),
            d_simparam.dr);

          // Only record if photon is not at the edge!!
          // This will be ignored anyways.
          if (iz < d_simparam.nz && ir < d_simparam.nr)
          {
            UINT32 addr = ir * d_simparam.nz + iz;

            if (addr != last_addr)
            {
#ifndef USE_TRUE_CACHE
              // Commit the weight drop to memory.
              if (last_ir < MAX_IR && last_iz < MAX_IZ)
              {
                // Write it to the shared memory.
                last_addr = last_ir * MAX_IZ + last_iz;
#ifdef USE_32B_ELEM_FOR_ARZ_SMEM
                // Use 32-bit atomicAdd.
                UINT32 oldval = atomicAdd(&A_rz_shared[last_addr], last_w);
                // Detect overflow.
                if (oldval >= d_simparam.A_rz_overflow)
                {
                  A_rz_overflow[last_addr % blockDim.x] = 1;
                }
#else
                // 64-bit atomic instruction
                AtomicAddULL_Shared(&A_rz_shared[last_addr], last_w);
#endif
              }
              else
#endif
              {
                // if (ir > 1024)
                {
                // Write it to the global memory directly.
                atomicAdd(&g_A_rz[last_addr], (UINT64)last_w);
                }
              }

              last_ir = ir; last_iz = iz;
              last_addr = addr;

              // Reset the last weight.
              last_w = 0;
            }

            // Accumulate to the last weight.
            last_w += (UINT32)(dwa * WEIGHT_SCALE);
          }
        }
        //>>>>>>>>> end of Drop()

        Spin(d_layerspecs[photon_layer].g,
          &photon_ux, &photon_uy, &photon_uz,
          &rnd_x, &rnd_a);
      }

      /***********************************************************
      *  >>>>>>>>> Roulette()
      *  If the photon weight is small, the photon packet tries
      *  to survive a roulette.
      ****/
      if (photon_w < WEIGHT)
      {
        FLOAT rand = rand_MWC_co(&rnd_x, &rnd_a);

        if (photon_w != MCML_FP_ZERO && rand < CHANCE)
          // This photon survives the roulette.
        {

          photon_w *= (FP_ONE / CHANCE);
        }
        // This photon is terminated.
        else if (atomicSub(d_state.n_photons_left, 1) > gridDim.x*blockDim.x)
        {
          // Launch a new photon.
          LaunchPhoton(&photon_x, &photon_y, &photon_z,
            &photon_ux, &photon_uy, &photon_uz,
            &photon_w, &photon_layer);
        }
        else
        {
          // No need to process any more photons.
          is_active = 0;
        }
      }
    }

    //////////////////////////////////////////////////////////////////////////

#if !defined(USE_TRUE_CACHE) && defined(USE_32B_ELEM_FOR_ARZ_SMEM)
    if (ignoreAdetection == 0)
    {
      // Enter a phase of handling overflow in A_rz_shared.
      __syncthreads();

      if (A_rz_overflow[threadIdx.x])
      {
        // Flush all elements I am responsible for to the global memory.
        for (int i = threadIdx.x; i < MAX_IR*MAX_IZ; i += blockDim.x)
        {
          Flush_Arz(g_A_rz, A_rz_shared, i);
          A_rz_shared[i] = 0;
        }
        // Reset the flag.
        A_rz_overflow[threadIdx.x] = 0;
      }

      __syncthreads();
    }
#endif

    //////////////////////////////////////////////////////////////////////
  } // end of the main loop

  __syncthreads();

  if (ignoreAdetection == 0)
  {
    // Commit the last weight drop.
    // NOTE: last_w == 0 if inactive.
    if (last_w > 0)
    {
#if 1
      // Commit to the global memory directly.
      atomicAdd(&g_A_rz[last_addr], last_w);
#else
      // Commit to A_rz_shared.
      UINT32 s_addr = last_ir * MAX_IZ + last_iz;
      atomicAdd(&g_A_rz[s_addr], last_w);
#endif
    }
  }

  //////////////////////////////////////////////////////////////////////////

#if 1
#ifndef USE_TRUE_CACHE
  if (ignoreAdetection == 0)
  {
    // Flush A_rz_shared to the global memory.
    for (int i = threadIdx.x; i < MAX_IR*MAX_IZ; i += blockDim.x)
    {
      Flush_Arz(g_A_rz, A_rz_shared, i);
    }
  }
#endif
#endif

  //////////////////////////////////////////////////////////////////////////

  // Save the thread state to the global memory.
  SaveThreadState(&d_state, &tstates, photon_x, photon_y, photon_z,
    photon_ux, photon_uy, photon_uz, photon_w,
    photon_layer,
    rnd_x, rnd_a,
    is_active);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

__global__ void sum_A_rz(UINT64 *g_A_rz)
{
  UINT64 sum;

  int n_elems = d_simparam.nz * d_simparam.nr;
  int base_ofst, ofst;

  for (base_ofst = blockIdx.x * blockDim.x + threadIdx.x;
      base_ofst < n_elems; base_ofst += blockDim.x * gridDim.x)
  {
    sum = 0;
    ofst = base_ofst;
#pragma unroll
    for (int i = 0; i < N_A_RZ_COPIES; ++i)
    {
      sum += g_A_rz[ofst];
      ofst += n_elems;
    }
    g_A_rz[base_ofst] = sum;
  }
}

#endif  // _GPUMCML_KERNEL_CU_

