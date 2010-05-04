/***********************************************************
 *  CUDA GPU version of MCML
 *  Kernel code for Monte Carlo simulation of photon
 *	distribution in multi-layered turbid media.
 *  Using shared memory for high light dose region (photon beam)
 *  May 1, 2009
 ****/

#ifndef _CUDAMCML_KERNEL_CU_
#define _CUDAMCML_KERNEL_CU_

#include "cudamcml_kernel.h"
#include "cudamcml_rng.cu"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

__device__ void LaunchPhoton(FLOAT *x, FLOAT *y, FLOAT *z,
        FLOAT *ux, FLOAT *uy, FLOAT *uz,
        FLOAT *w, FLOAT *sleft, UINT32 *layer)
{
    *x = *y = *z = MCML_FP_ZERO;
    *ux = *uy = MCML_FP_ZERO;
    *uz = FP_ONE;
    *w = d_simparam.init_photon_w;
    *sleft = MCML_FP_ZERO;
    *layer = 1;
}

// Initialize per-thread states (except random number seeds).
__global__ void InitThreadState(GPUThreadStates tstates)
{
    FLOAT photon_x, photon_y, photon_z;
    FLOAT photon_ux, photon_uy, photon_uz;
    FLOAT photon_w, photon_sleft;
    UINT32 photon_layer;

    LaunchPhoton(&photon_x, &photon_y, &photon_z,
            &photon_ux, &photon_uy, &photon_uz,
            &photon_w, &photon_sleft, &photon_layer);

    UINT32 tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

    tstates.photon_x[tid] = photon_x;
    tstates.photon_y[tid] = photon_y;
    tstates.photon_z[tid] = photon_z;
    tstates.photon_ux[tid] = photon_ux;
    tstates.photon_uy[tid] = photon_uy;
    tstates.photon_uz[tid] = photon_uz;
    tstates.photon_w[tid] = photon_w;
    tstates.photon_sleft[tid] = photon_sleft;
    tstates.photon_layer[tid] = photon_layer;

    tstates.is_active[tid] = 1;
}

__device__ void SaveThreadState(SimState *d_state, GPUThreadStates *tstates,
        FLOAT photon_x, FLOAT photon_y, FLOAT photon_z,
        FLOAT photon_ux, FLOAT photon_uy, FLOAT photon_uz,
        FLOAT photon_w, FLOAT photon_sleft, UINT32 photon_layer,
#ifdef USE_MT_RNG
        UINT64 rnd_x, UINT32 rnd_a,
#else
        UINT64 rnd_s1, UINT32 rnd_s2, UINT32 rnd_s3,
#endif
        UINT32 is_active)
{
    UINT32 tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

#ifdef USE_MT_RNG
    d_state->x[tid] = rnd_x;
    d_state->a[tid] = rnd_a;
#else
    d_state->s1[tid] = rnd_s1;
    d_state->s2[tid] = rnd_s2;
    d_state->s3[tid] = rnd_s3;
#endif

    tstates->photon_x[tid] = photon_x;
    tstates->photon_y[tid] = photon_y;
    tstates->photon_z[tid] = photon_z;
    tstates->photon_ux[tid] = photon_ux;
    tstates->photon_uy[tid] = photon_uy;
    tstates->photon_uz[tid] = photon_uz;
    tstates->photon_w[tid] = photon_w;
    tstates->photon_sleft[tid] = photon_sleft;
    tstates->photon_layer[tid] = photon_layer;

    tstates->is_active[tid] = is_active;
}

__device__ void RestoreThreadState(SimState *d_state, GPUThreadStates *tstates,
        FLOAT *photon_x, FLOAT *photon_y, FLOAT *photon_z,
        FLOAT *photon_ux, FLOAT *photon_uy, FLOAT *photon_uz,
        FLOAT *photon_w, FLOAT *photon_sleft, UINT32 *photon_layer,
#ifdef USE_MT_RNG
        UINT64 *rnd_x, UINT32 *rnd_a,
#else
        UINT32 *rnd_s1, UINT32 *rnd_s2, UINT32 *rnd_s3,
#endif
        UINT32 *is_active)
{
    UINT32 tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

#ifdef USE_MT_RNG
    *rnd_x = d_state->x[tid];
    *rnd_a = d_state->a[tid];
#else
    *rnd_s1 = d_state->s1[tid];
    *rnd_s2 = d_state->s2[tid];
    *rnd_s3 = d_state->s3[tid];
#endif

    *photon_x = tstates->photon_x[tid];
    *photon_y = tstates->photon_y[tid];
    *photon_z = tstates->photon_z[tid];
    *photon_ux = tstates->photon_ux[tid];
    *photon_uy = tstates->photon_uy[tid];
    *photon_uz = tstates->photon_uz[tid];
    *photon_w = tstates->photon_w[tid];
    *photon_sleft = tstates->photon_sleft[tid];
    *photon_layer = tstates->photon_layer[tid];

    *is_active = tstates->is_active[tid];
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Add an UINT32eger to an UINT64
// using CUDA Compute Capability 1.1
__device__ void AtomicAddULL(UINT64* address, UINT32 add)
{
#ifdef USE_ATOMICADD_ULL_CC1
    if (atomicAdd((UINT32*)address,add) +add < add)
    {
        atomicAdd(((UINT32*)address)+1, 1U);
    }
#else
    atomicAdd(address, (UINT64)add);
#endif
}

/***********************************************************
 *	>>>>>>>>> StepSizeInTissue()
 * 	Pick a step size for a photon packet when it is in tissue.
 *	If the member sleft is zero, make a new step size
 *	with: -log(rnd)/(mua+mus).
 *	Otherwise, pick up the leftover in sleft.
 ****/
__device__ void ComputeStepSize(UINT32 layer,
        FLOAT *s_ptr, FLOAT *sleft_ptr,
#ifdef USE_MT_RNG
        UINT64 *rnd_x, UINT32 *rnd_a)
#else
        UINT32 *rnd_s1, UINT32 *rnd_s2, UINT32 *rnd_s3)
#endif
{
    // Make a new step if no leftover.
    FLOAT s = *sleft_ptr;
    if (s == MCML_FP_ZERO)
    {
#ifdef USE_MT_RNG
        FLOAT rand = rand_MWC_oc(rnd_x, rnd_a);
#else
        FLOAT rand = Rand_Taus_nz(rnd_s1, rnd_s2, rnd_s3);
#endif
        s = -__logf(rand);
    }

    *s_ptr = s * d_layerspecs[layer].rmuas;
    *sleft_ptr = MCML_FP_ZERO;
}

/***********************************************************
 *  >>>>>>>>> HitBoundary()
 *	Check if the step will hit the boundary.
 *	Return 1 if hit boundary.
 *	Return 0 otherwise.
 *
 * 	If the projected step hits the boundary, the members
 *	s and sleft of Photon_Ptr are updated.
 ****/
__device__ int HitBoundary(UINT32 layer, FLOAT z, FLOAT uz,
        FLOAT *s_ptr, FLOAT *sleft_ptr)
{
    FLOAT dl_b; /* step size to boundary. */

    /* Distance to the boundary. */
    FLOAT z_bound = (uz > MCML_FP_ZERO) ?
        d_layerspecs[layer].z1 : d_layerspecs[layer].z0;
    dl_b = __fdividef(z_bound - z, uz);     // dl_b > 0

    FLOAT s = *s_ptr;
    UINT32 hit_boundary = (uz != MCML_FP_ZERO) && (s > dl_b);
    if (hit_boundary)
    {
        /* not horizontal & crossing. */

        // No need to multiply by (mua + mus), as it is later
        // divided by (mua + mus) anyways (in the original version).
        *sleft_ptr = (s - dl_b) * d_layerspecs[layer].muas;
        *s_ptr = dl_b;
    }

    return hit_boundary;
}

//***********************************************************
// >>>>>>>>> Hop()
// Move the photon s away in the current layer of medium.
__device__ void Hop(FLOAT s, FLOAT ux, FLOAT uy, FLOAT uz,
        FLOAT *x, FLOAT *y, FLOAT *z)
{
    *x += s * ux;
    *y += s * uy;
    *z += s * uz;
}

/***********************************************************
 *	>>>>>>>>> UltraFast version()
 *	>>>>>>>>> Reduced divergence
*/
__device__ void FastReflectTransmit(FLOAT x, FLOAT y, SimState *d_state_ptr,
        FLOAT *ux, FLOAT *uy, FLOAT *uz,
        UINT32 *layer, FLOAT* w,
#ifdef USE_MT_RNG
        UINT64 *rnd_x, UINT32 *rnd_a)
#else
        UINT32 *rnd_s1, UINT32 *rnd_s2, UINT32 *rnd_s3)
#endif
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

#ifdef USE_MT_RNG
        FLOAT rand = rand_MWC_co(rnd_x, rnd_a);
#else
        FLOAT rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
#endif
        if (rFresnel < rand)
        {
            // The move is to transmit.
            *layer = new_layer;

            // Let's do these even if the photon is dead.
            *ux *= ni_nt;
            *uy *= ni_nt;
            // Is this faster?
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

                AtomicAddULL(&ra_arr[ia * d_simparam.nr + ir],
                        (UINT32)(*w * WEIGHT_SCALE));

                // Kill the photon.
                *w = MCML_FP_ZERO;
            }
        }
    }
}

/***********************************************************
 *	>>>>>>>>> Spin()
 *  Choose a new direction for photon propagation by
 *	sampling the polar deflection angle theta and the
 *	azimuthal angle psi.
 *
 *	Note:
 *  	theta: 0 - pi so sin(theta) is always positive
 *  	feel free to use sqrtf() for cos(theta).
 *
 *  	psi:   0 - 2pi
 *  	for 0-pi  sin(psi) is +
 *  	for pi-2pi sin(psi) is -
 ****/
__device__ void Spin(FLOAT g, FLOAT *ux, FLOAT *uy, FLOAT *uz,
#ifdef USE_MT_RNG
        UINT64 *rnd_x, UINT32 *rnd_a)
#else
        UINT32 *rnd_s1, UINT32 *rnd_s2, UINT32 *rnd_s3)
#endif
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

#ifdef USE_MT_RNG
    rand = rand_MWC_co(rnd_x, rnd_a);
#else
    rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
#endif
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
#ifdef USE_MT_RNG
    rand = rand_MWC_co(rnd_x, rnd_a);
#else
    rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
#endif
    psi = FP_TWO * PI_const * rand;
    __sincosf(psi, &sinp, &cosp);

    FLOAT stcp = sint * cosp;
    FLOAT stsp = sint * sinp;

    last_ux = *ux;
    last_uy = *uy;
    last_uz = *uz;

    // DAVID
#if 0
    if (fabsf(last_uz) > COSZERO)
    {
        /* normal incident. */
        *ux = sint * cosp;
        *uy = sint * sinp;
        SIGN = ((last_uz) >= 0.0f ? 1.0f : -1.0f);
        /* SIGN() is faster than division. */
        //cost= //cost*SIGN; //copysignf (cost, last_uz);
        //		*uz =copysignf (cost, last_uz*cost);;  //WHY do we need to multiply by cost to get right ans??
        *uz = cost * SIGN;

    } else { /* regular incident. */
        temp = rsqrtf(1.0f - last_uz * last_uz);
        *ux = sint * (last_ux * last_uz * cosp - last_uy * sinp) * temp
            + last_ux * cost;
        *uy = sint * (last_uy * last_uz * cosp + last_ux * sinp) * temp
            + last_uy * cost;
        *uz = __fdividef(-sint * cosp, temp) + last_uz * cost;
    }
#else
    if (fabsf(last_uz) > COSZERO)
    {
        *ux = stcp;
        *uy = stsp;
        SIGN = ((last_uz) >= MCML_FP_ZERO ? FP_ONE : -FP_ONE);
        *uz = cost * SIGN;
    }
    else
    {
        temp = rsqrtf(FP_ONE - last_uz * last_uz);
        *ux = (stcp * last_ux * last_uz - stsp * last_uy) * temp
            + last_ux * cost;
        *uy = (stcp * last_uy * last_uz + stsp * last_ux) * temp
            + last_uy * cost;
        *uz = __fdividef(-stcp, temp) + last_uz * cost;
    }
#endif
}

/*****************************************************************************
 *
 * Flush the element at offset <shared_addr> of A_rz in shared memory (s_A_rz)
 * to the global memory (g_A_rz). <s_A_rz> is of dimension MAX_IR x MAX_IZ.
 *
 ****************************************************************************/

__device__ void Flush_Arz(UINT64 *g_A_rz,
        UINT64 *s_A_rz, UINT32 shared_addr)
{
    UINT32 ir = shared_addr / MAX_IZ;
    UINT32 iz = shared_addr - ir * MAX_IZ;
    UINT32 global_addr = ir * d_simparam.nz + iz;

    atomicAdd(&g_A_rz[global_addr], s_A_rz[shared_addr]);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <int ignoreAdetection>
__global__ void MCMLKernel(SimState d_state, GPUThreadStates tstates)
{
    // photon structure
    FLOAT photon_x, photon_y ,photon_z;
    FLOAT photon_ux, photon_uy, photon_uz;
    FLOAT photon_w, photon_sleft;
    UINT32 photon_layer;

    // random number seeds
#ifdef USE_MT_RNG
    UINT64 rnd_x;
    UINT32 rnd_a;
#else
    UINT32 rnd_s1, rnd_s2, rnd_s3;
#endif

    // is this thread active?
    UINT32 is_active;

    // Restore the thread state from global memory.
    RestoreThreadState(&d_state, &tstates,
            &photon_x, &photon_y, &photon_z,
            &photon_ux, &photon_uy, &photon_uz,
            &photon_w, &photon_sleft, &photon_layer,
#ifdef USE_MT_RNG
            &rnd_x, &rnd_a,
#else
            &rnd_s1, &rnd_s2, &rnd_s3,
#endif
            &is_active);

    //////////////////////////////////////////////////////////////////////////

    // Coalesce consecutive weight drops to the same address.
    UINT32 last_w = 0;
    UINT32 last_ir = 0, last_iz = 0, last_addr = 0;

    //////////////////////////////////////////////////////////////////////////

#ifndef USE_TRUE_CACHE
    // Cache the frequently acessed region of A_rz in the shared memory.
    __shared__ UINT64 A_rz_shared[MAX_IR*MAX_IZ];

    if (ignoreAdetection == 0)
    {
        // Clear the cache.
        for (int i = threadIdx.x; i < MAX_IR*MAX_IZ;
                i += NUM_THREADS_PER_BLOCK) A_rz_shared[i] = 0;
        __syncthreads();
    }
#endif

    //////////////////////////////////////////////////////////////////////////

    // Get the copy of A_rz (in the global memory) this thread writes to.
    UINT64 *g_A_rz = d_state.A_rz; 
    //    + (blockIdx.x % N_A_RZ_COPIES) * (d_simparam.nz * d_simparam.nr);

    //////////////////////////////////////////////////////////////////////////

    for (int iIndex = 0; iIndex < NUM_STEPS; ++iIndex)
    {
        // Only process photon if the thread is active.
        if (is_active)
        {
            FLOAT photon_s;     // current step size

            //>>>>>>>>> StepSizeInTissue()
            ComputeStepSize(photon_layer, &photon_s, &photon_sleft,
#ifdef USE_MT_RNG
                    &rnd_x, &rnd_a);
#else
                    &rnd_s1, &rnd_s2, &rnd_s3);
#endif

            //>>>>>>>>> HitBoundary()
            UINT32 photon_hit = HitBoundary(photon_layer,
                    photon_z, photon_uz, &photon_s, &photon_sleft);

            Hop(photon_s, photon_ux, photon_uy, photon_uz,
                    &photon_x, &photon_y, &photon_z);

            if (photon_hit)
            {
                FastReflectTransmit(photon_x, photon_y, &d_state,
                        &photon_ux, &photon_uy, &photon_uz,
                        &photon_layer, &photon_w,
#ifdef USE_MT_RNG
                        &rnd_x, &rnd_a);
#else
                        &rnd_s1, &rnd_s2, &rnd_s3);
#endif
            }
            else
            {
                //>>>>>>>>> Drop()
                FLOAT dwa = photon_w * d_layerspecs[photon_layer].mua_muas;
                photon_w -= dwa;

                // DAVID
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
                        UINT32 addr = ir * MAX_IZ + iz;

                        if (addr != last_addr)
                        {
#ifndef USE_TRUE_CACHE
                            // Commit the weight drop to memory.
                            if (last_addr < MAX_IR * MAX_IZ)
                            {
                                // Write it to the shared memory.
                                AtomicAddULL(&A_rz_shared[last_addr], last_w);
                            }
                            else
#endif
                            {
                                // Write it to the global memory directly.
                                last_addr = last_ir * d_simparam.nz + last_iz;
                                atomicAdd(&g_A_rz[last_addr],
                                        (UINT64)last_w);
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
                //>>>>>>>>> Drop()

                Spin(d_layerspecs[photon_layer].g,
                        &photon_ux, &photon_uy, &photon_uz,
#ifdef USE_MT_RNG
                        &rnd_x, &rnd_a);
#else
                        &rnd_s1, &rnd_s2, &rnd_s3);
#endif
            }

            /***********************************************************
             *  >>>>>>>>> Roulette()
             * The photon weight is small, and the photon packet tries
             *  to survive a roulette.
             ****/
            if (photon_w < WEIGHT)
            {
#ifdef USE_MT_RNG
                FLOAT rand = rand_MWC_co(&rnd_x, &rnd_a);
#else
                FLOAT rand = Rand_Taus(&rnd_s1, &rnd_s2, &rnd_s3);
#endif
                if (photon_w != MCML_FP_ZERO && rand < CHANCE)
                {
                    // Survive the roulette.
                    photon_w *= (FP_ONE / CHANCE);
                }
                // This photon dies.
                else if (atomicSub(d_state.n_photons_left, 1) > NUM_THREADS)
                {
                    // Launch a new photon.
                    LaunchPhoton(&photon_x, &photon_y, &photon_z,
                            &photon_ux, &photon_uy, &photon_uz,
                            &photon_w, &photon_sleft, &photon_layer);
                }
                else
                {
                    // No need to process any more photons.
                    // Although inactive, this thread still needs to
                    // participate in overflow handling.
                    is_active = 0;
                }
            }
        }

        //////////////////////////////////////////////////////////////////////
    } // end of the main loop

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////

    if (ignoreAdetection == 0)
    {
        // Commit the last weight drop to the global memory directly.
        // NOTE: last_w == 0 if inactive.
        if (last_w > 0)
        {
            UINT32 global_addr = last_ir * d_simparam.nz + last_iz;
            AtomicAddULL(&g_A_rz[global_addr], last_w);
        }

#ifndef USE_TRUE_CACHE
        // Flush A_rz_shared to the global memory.
        for (int i = threadIdx.x; i < MAX_IR*MAX_IZ;
                i += NUM_THREADS_PER_BLOCK)
        {
            Flush_Arz(g_A_rz, A_rz_shared, i);
        }
#endif
    }

    //////////////////////////////////////////////////////////////////////////

    // Save the thread state to the global memory.
    SaveThreadState(&d_state, &tstates, photon_x, photon_y, photon_z,
            photon_ux, photon_uy, photon_uz, photon_w, photon_sleft,
            photon_layer,
#ifdef USE_MT_RNG
            rnd_x, rnd_a,
#else
            rnd_s1, rnd_s2, rnd_s3,
#endif
            is_active);
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//__global__ void sum_A_rz(UINT64 *g_A_rz)
//{
//    UINT64 sum;
//
//    int n_elems = d_simparam.nz * d_simparam.nr;
//    int base_ofst, ofst;
//
//    for (base_ofst = blockIdx.x * blockDim.x + threadIdx.x;
//            base_ofst < n_elems; base_ofst += blockDim.x * gridDim.x)
//    {
//        sum = 0;
//        ofst = base_ofst;
//#pragma unroll
//        for (int i = 0; i < N_A_RZ_COPIES; ++i)
//        {
//            sum += g_A_rz[ofst];
//            ofst += n_elems;
//        }
//        g_A_rz[base_ofst] = sum;
//    }
//}

#endif  // _CUDAMCML_KERNEL_CU_

