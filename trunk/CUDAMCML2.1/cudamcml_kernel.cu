/***********************************************************
 *  CUDA GPU version of MCML
 *  Kernel code for Monte Carlo simulation of photon
 *	distribution in multi-layered turbid media.
 *  Using shared memory for high light dose region (photon beam)
 *  October 28, 2009
 ****/

#ifndef _CUDAMCML_KERNEL_CU_
#define _CUDAMCML_KERNEL_CU_

#include "cudamcml_kernel.h"
#include "cudamcml_rng.cu"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

/*****************************************************************************
 *
 * This routine computes the maximum element value of A_rz in shared memory,
 * that indicates an imminent overflow.
 *
 * This MAX_OVERFLOW is MAX_UINT32 - MAX(dwa) * NUM_THREADS_PER_BLOCK.
 *
 * All we really need to compute is
 *      MAX(dwa) <= WEIGHT_SCALE * <init_photon_w> * MAX( mua/(mua+mus) )
 *
 * We have to be accurate in this bound because if we assume that
 *      MAX(dwa) = WEIGHT_SCALE,
 * MAX_OVERFLOW can be small if WEIGHT_SCALE is large, like 12000000.
 *
 * <n_layers> is the length of <layers>, excluding the top and bottom layers.
 *
 ****************************************************************************/

unsigned int compute_Arz_overflow_count(float init_photon_w,
        LayerStruct *layers, int n_layers)
{
    // Determine the largest mua/(mua+mus) over all layers.
    double max_mua_muas = 0;
    for (int i = 1; i <= n_layers; ++i)
    {
        double mua_muas = layers[i].mua * layers[i].mutr;
        if (max_mua_muas < mua_muas) max_mua_muas = mua_muas;
    }

    // Determine max_dwa.
    unsigned int max_dwa =
        (unsigned int)(init_photon_w * max_mua_muas * (float)WEIGHT_SCALE);
    // Just to be safe.
    ++max_dwa;

    unsigned int max_overflow = 0xFFFFFFFF - max_dwa * NUM_THREADS_PER_BLOCK;

    return max_overflow;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

__device__ void LaunchPhoton(float *x, float *y, float *z,
        float *ux, float *uy, float *uz,
        float *w, float *sleft, unsigned int *layer)
{
    *x = *y = *z = 0.0F;
    *ux = *uy = 0.0F;
    *uz = 1.0F;
    *w = d_simparam.init_photon_w;
    *sleft = 0.0F;
    *layer = 1;
}

// Initialize per-thread states (except random number seeds).
__global__ void InitThreadState(GPUThreadStates tstates)
{
    float photon_x, photon_y, photon_z;
    float photon_ux, photon_uy, photon_uz;
    float photon_w, photon_sleft;
    unsigned int photon_layer;

    LaunchPhoton(&photon_x, &photon_y, &photon_z,
            &photon_ux, &photon_uy, &photon_uz,
            &photon_w, &photon_sleft, &photon_layer);

    unsigned int tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

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
        float photon_x, float photon_y, float photon_z,
        float photon_ux, float photon_uy, float photon_uz,
        float photon_w, float photon_sleft, unsigned int photon_layer,
#ifdef USE_MT_RNG
        unsigned long long rnd_x, unsigned int rnd_a,
#else
        unsigned long long rnd_s1, unsigned int rnd_s2, unsigned int rnd_s3,
#endif
        unsigned int is_active)
{
    unsigned int tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

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
        float *photon_x, float *photon_y, float *photon_z,
        float *photon_ux, float *photon_uy, float *photon_uz,
        float *photon_w, float *photon_sleft, unsigned int *photon_layer,
#ifdef USE_MT_RNG
        unsigned long long *rnd_x, unsigned int *rnd_a,
#else
        unsigned int *rnd_s1, unsigned int *rnd_s2, unsigned int *rnd_s3,
#endif
        unsigned int *is_active)
{
    unsigned int tid = blockIdx.x * NUM_THREADS_PER_BLOCK + threadIdx.x;

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

// Add an unsigned integer to an unsigned long long
// using CUDA Compute Capability 1.1
__device__ void AtomicAddULL(unsigned long long* address, unsigned int add)
{
    if (atomicAdd((unsigned int*)address,add) +add < add)
        atomicAdd(((unsigned int*)address)+1, 1u);
}

/***********************************************************
 *	>>>>>>>>> StepSizeInTissue()
 * 	Pick a step size for a photon packet when it is in tissue.
 *	If the member sleft is zero, make a new step size
 *	with: -log(rnd)/(mua+mus).
 *	Otherwise, pick up the leftover in sleft.
 ****/
__device__ void ComputeStepSize(unsigned int layer,
        float *s_ptr, float *sleft_ptr,
#ifdef USE_MT_RNG
        unsigned long long *rnd_x, unsigned int *rnd_a)
#else
        unsigned int *rnd_s1, unsigned int *rnd_s2, unsigned int *rnd_s3)
#endif
{
    // Make a new step if no leftover.
    float s = *sleft_ptr;
    if (s == 0.0F)
    {
#ifdef USE_MT_RNG
        float rand = rand_MWC_oc(rnd_x, rnd_a);
#else
        float rand = Rand_Taus_nz(rnd_s1, rnd_s2, rnd_s3);
#endif
        s = -__logf(rand);
    }

    *s_ptr = s * d_layerspecs[layer].rmuas;
    *sleft_ptr = 0.0F;
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
__device__ int HitBoundary(unsigned int layer, float z, float uz,
        float *s_ptr, float *sleft_ptr)
{
    float dl_b; /* step size to boundary. */

    /* Distance to the boundary. */
    float z_bound = (uz > 0.0F) ?
        d_layerspecs[layer].z1 : d_layerspecs[layer].z0;
    dl_b = __fdividef(z_bound - z, uz);     // dl_b > 0

    float s = *s_ptr;
    unsigned int hit_boundary = (uz != 0.0F) && (s > dl_b);
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
__device__ void Hop(float s, float ux, float uy, float uz,
        float *x, float *y, float *z)
{
    *x += s * ux;
    *y += s * uy;
    *z += s * uz;
}

/***********************************************************
 *	>>>>>>>>> UltraFast version()
 *	>>>>>>>>> Reduced divergence
*/
__device__ void FastReflectTransmit(float x, float y, SimState *d_state_ptr,
        float *ux, float *uy, float *uz,
        unsigned int *layer, float* w,
#ifdef USE_MT_RNG
        unsigned long long *rnd_x, unsigned int *rnd_a)
#else
        unsigned int *rnd_s1, unsigned int *rnd_s2, unsigned int *rnd_s3)
#endif
{
    /* Collect all info that depend on the sign of "uz". */
    float cos_crit;
    unsigned int new_layer;
    if (*uz > 0.0F)
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
    float ca1 = fabsf(*uz);

    // The default move is to reflect.
    *uz = -(*uz);

    // Moving this check down to "RFresnel = 0.0F" slows down the
    // application, possibly because every thread is forced to do
    // too much.
    if (ca1 > cos_crit)
    {
        /* Compute the Fresnel reflectance. */

        // incident and transmit refractive index
        float ni = d_layerspecs[(*layer)].n;
        float nt = d_layerspecs[new_layer].n;
        float ni_nt = __fdividef(ni, nt);   // reused later

        float sa1 = sqrtf(1.0F-ca1*ca1);
        float sa2 = fminf(ni_nt * sa1, 1.0F);
        if (ca1 > COSZERO) sa2 = sa1;
        float uz1 = sqrtf(1.0F-sa2*sa2);    // uz1 = ca2

        float ca1ca2 = ca1 * uz1;
        float sa1sa2 = sa1 * sa2;
        float sa1ca2 = sa1 * uz1;
        float ca1sa2 = ca1 * sa2;

        float cam = ca1ca2 + sa1sa2; /* c- = cc + ss. */
        float sap = sa1ca2 + ca1sa2; /* s+ = sc + cs. */
        float sam = sa1ca2 - ca1sa2; /* s- = sc - cs. */

        float rFresnel = __fdividef(sam, sap*cam);
        rFresnel *= rFresnel;
        rFresnel *= (ca1ca2*ca1ca2 + sa1sa2*sa1sa2);

        // Hope "uz1" is very close to "ca1".
        if (ca1 > COSZERO) rFresnel = 0.0F;
        // In this case, we do not care if "uz1" is exactly 0.
        if (ca1 < COSNINETYDEG || sa2 == 1.0F) rFresnel = 1.0F;

#ifdef USE_MT_RNG
        float rand = rand_MWC_co(rnd_x, rnd_a);
#else
        float rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
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
                float uz2 = *uz;
                unsigned long long *ra_arr = d_state_ptr->Tt_ra;
                if (*layer == 0)
                {
                    // diffuse reflectance
                    uz2 = -uz2;
                    ra_arr = d_state_ptr->Rd_ra;
                }

                unsigned int ia = acosf(uz2) * 2.0F * RPI * d_simparam.na;
                unsigned int ir = __fdividef(sqrtf(x*x+y*y), d_simparam.dr);
                if (ir >= d_simparam.nr) ir = d_simparam.nr - 1;

#ifdef USE_ATOMICADD_ULL_CC1
                AtomicAddULL(&ra_arr[ia * d_simparam.nr + ir],
                        (unsigned int)(*w * WEIGHT_SCALE));
#else
                atomicAdd(&ra_arr[ia * d_simparam.nr + ir],
                        (unsigned long long)(*w * WEIGHT_SCALE));
#endif

                // Kill the photon.
                *w = 0.0F;
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
__device__ void Spin(float g, float *ux, float *uy, float *uz,
#ifdef USE_MT_RNG
        unsigned long long *rnd_x, unsigned int *rnd_a)
#else
        unsigned int *rnd_s1, unsigned int *rnd_s2, unsigned int *rnd_s3)
#endif
{
    //const float COSZERO = 1.0f - 1.0E-6f; //NOTE: 1.0-1.0E-7 doesn't work in CUDA!!!
    //>>>>>>>>> Spin()
    float cost, sint; /* cosine and sine of the polar deflection angle theta. */
    float cosp, sinp; /* cosine and sine of the azimuthal angle psi. */
    float SIGN;
    float temp;
    float last_ux = *ux;
    float last_uy = *uy;
    float last_uz = *uz;

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
    float rand = rand_MWC_co(rnd_x, rnd_a);
#else
    float rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
#endif

    cost = 2.0F * rand - 1.0F;
    temp = __fdividef((1.0f - g * g), 1.0F + g*cost);
    if (g != 0.0F)
    {
        cost = __fdividef(1.0F + g * g - temp*temp, 2.0F * g);
        cost = fmaxf(cost, -1.0F);
        cost = fminf(cost, 1.0F);
    }

    sint = sqrtf(1.0f - cost * cost);
    /* sqrtf() is faster than sin(). */

    /* spin psi 0-2pi. */
#ifdef USE_MT_RNG
    rand = rand_MWC_co(rnd_x, rnd_a);
#else
    rand = Rand_Taus(rnd_s1, rnd_s2, rnd_s3);
#endif
    float psi = 2.0F * PI_const * rand;
    __sincosf(psi, &sinp, &cosp);

    if (fabsf(last_uz) > COSZERO) { /* normal incident. */
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
}

/*****************************************************************************
 *
 * Flush the element at offset <shared_addr> of A_rz in shared memory (s_A_rz)
 * to the global memory (g_A_rz). <s_A_rz> is of dimension MAX_IR x MAX_IZ.
 *
 ****************************************************************************/

__device__ void Flush_Arz(unsigned long long *g_A_rz, unsigned int *s_A_rz,
        unsigned int shared_addr)
{
    unsigned int ir = shared_addr / MAX_IZ;
    unsigned int iz = shared_addr - ir * MAX_IZ;
    unsigned int global_addr = ir * d_simparam.nz + iz;

#if USE_ATOMICADD_ULL_CC1
    AtomicAddULL(&g_A_rz[global_addr], s_A_rz[shared_addr]);
#else
    atomicAdd(&g_A_rz[global_addr], (unsigned long long)s_A_rz[shared_addr]);
#endif
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

template <int ignoreAdetection>
__global__ void MCMLKernel(SimState d_state, GPUThreadStates tstates)
{
    // photon structure
    float photon_x, photon_y ,photon_z;
    float photon_ux, photon_uy, photon_uz;
    float photon_w, photon_sleft;
    unsigned int photon_layer;

    // random number seeds
#ifdef USE_MT_RNG
    unsigned long long rnd_x;
    unsigned int rnd_a;
#else
    unsigned int rnd_s1, rnd_s2, rnd_s3;
#endif

    // is this thread active?
    unsigned int is_active;

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
    unsigned int last_w = 0;
    unsigned int last_ir = 0, last_iz = 0;

    //////////////////////////////////////////////////////////////////////////

    // Cache the frequently acessed region of A_rz in the shared memory.
    __shared__ unsigned int A_rz_shared[MAX_IR*MAX_IZ];

    if (ignoreAdetection == 0)
    {
        // Clear the cache.
        for (int i = threadIdx.x; i < MAX_IR*MAX_IZ;
                i += NUM_THREADS_PER_BLOCK) A_rz_shared[i] = 0;
    }

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
    __shared__ unsigned int A_rz_overflow[NUM_THREADS_PER_BLOCK];

    if (ignoreAdetection == 0)
    {
        // Clear the flags.
        A_rz_overflow[threadIdx.x] = 0;
    }

    //////////////////////////////////////////////////////////////////////////

    for (int iIndex = 0; iIndex < NUM_STEPS; ++iIndex)
    {
        // Only process photon if the thread is active.
        if (is_active)
        {
            float photon_s;     // current step size

            //>>>>>>>>> StepSizeInTissue()
            ComputeStepSize(photon_layer, &photon_s, &photon_sleft,
#ifdef USE_MT_RNG
                    &rnd_x, &rnd_a);
#else
                    &rnd_s1, &rnd_s2, &rnd_s3);
#endif

            //>>>>>>>>> HitBoundary()
            unsigned int photon_hit = HitBoundary(photon_layer,
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
                unsigned int hit_edge = 0;

                // automatic __float2uint_rz
                unsigned int iz = __fdividef(photon_z , d_simparam.dz);
                if (iz >= d_simparam.nz)
                {
                    hit_edge = 1;
                    iz = d_simparam.nz - 1;
                }

                // automatic __float2uint_rz
                unsigned int ir = __fdividef(
                            sqrtf(photon_x * photon_x + photon_y * photon_y),
                            d_simparam.dr);
                if (ir >= d_simparam.nr)
                {
                    hit_edge = 1;
                    ir = d_simparam.nr - 1;
                }

                float dwa = photon_w * d_layerspecs[photon_layer].mua_muas;
                photon_w -= dwa;

                //only record if photon is not at the edge!!
                //this will be ignored anyways.
                if (ignoreAdetection == 0 && !hit_edge)
                {
                    if (ir != last_ir || iz != last_iz)
                    {
                        // Commit the last weight drop to memory.
                        unsigned int addr;
                        if (last_ir < MAX_IR && last_iz < MAX_IZ)
                        {
                            // Write it to the shared memory.
                            addr = last_ir * MAX_IZ + last_iz;
                            unsigned int oldval =
                                atomicAdd(&A_rz_shared[addr], last_w);
                            // Detect overflow.
                            if (oldval >= d_simparam.A_rz_overflow)
                            {
                                A_rz_overflow[addr % NUM_THREADS_PER_BLOCK] = 1;
                            }
                        }
                        else
                        {
                            // Write it to the global memory directly.
                            addr = last_ir * d_simparam.nz + last_iz;
#if USE_ATOMICADD_ULL_CC1
                            AtomicAddULL(&d_state.A_rz[addr], last_w);
#else
                            atomicAdd(&d_state.A_rz[addr],
                                    (unsigned long long)last_w);
#endif
                        }

                        // Reset the last weight.
                        last_w = 0;
                    }

                    last_ir = ir; last_iz = iz;
                    // Accumulate to the last weight.
                    last_w += (unsigned int)(dwa * WEIGHT_SCALE);
                } //end if (!hit_edge)
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
                float rand = rand_MWC_co(&rnd_x, &rnd_a);
#else
                float rand = Rand_Taus(&rnd_s1, &rnd_s2, &rnd_s3);
#endif
                if (photon_w != 0.0F && rand < CHANCE)
                {
                    // Survive the roulette.
                    photon_w *= (1.0F / CHANCE);
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

        if (ignoreAdetection == 0)
        {
            // Enter a phase of handling overflow in A_rz_shared.
            __syncthreads();
            if (A_rz_overflow[threadIdx.x])
            {
                // Flush all elements I am responsible for to the global memory.
                for (unsigned int i = threadIdx.x; i < MAX_IR*MAX_IZ;
                        i += NUM_THREADS_PER_BLOCK)
                {
                    Flush_Arz(d_state.A_rz, A_rz_shared, i);
                    A_rz_shared[i] = 0;
                }
                A_rz_overflow[threadIdx.x] = 0; // reset the flag.
            }
            __syncthreads();
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
            unsigned int global_addr = last_ir * d_simparam.nz + last_iz;
#if USE_ATOMICADD_ULL_CC1
            AtomicAddULL(&d_state.A_rz[global_addr], last_w);
#else
            atomicAdd(&d_state.A_rz[global_addr], (unsigned long long)last_w);
#endif
        }

        // Flush A_rz_shared to the global memory.
        for (unsigned int i = threadIdx.x; i < MAX_IR*MAX_IZ;
                i += NUM_THREADS_PER_BLOCK)
        {
            Flush_Arz(d_state.A_rz, A_rz_shared, i);
        }
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

#endif  // _CUDAMCML_KERNEL_CU_

