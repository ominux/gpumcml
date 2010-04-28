/////////////////////////////////////////////////////////////
//
//		CUDA-based Monte Carlo simulation of photon migration in layered media (CUDAMCML).
//	
//			Some documentation is avialable for CUDAMCML and should have been distrbuted along 
//			with this source code. If that is not the case: Documentation, source code and executables
//			for CUDAMCML are available for download on our webpage:
//			http://www.atomic.physics.lu.se/Biophotonics
//			or, directly
//			http://www.atomic.physics.lu.se/fileadmin/atomfysik/Biophotonics/Software/CUDAMCML.zip
//
//			We encourage the use, and modifcation of this code, and hope it will help 
//			users/programmers to utilize the power of GPGPU for their simulation needs. While we
//			don't have a scientifc publication describing this code, we would very much appreciate
//			if you cite our original GPGPU Monte Carlo letter (on which CUDAMCML is based) if you 
//			use this code or derivations thereof for your own scientifc work:
//			E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics processing
//			units for high-speed Monte Carlo simulations of photon migration", Journal of Biomedical Optics
//			Letters, 13(6) 060504 (2008).
//
//			To compile and run this code, please visit www.nvidia.com and download the necessary 
//			CUDA Toolkit and SKD. We also highly recommend the Visual Studio wizard 
//			(available at:http://forums.nvidia.com/index.php?showtopic=69183) 
//			if you use Visual Studio 2005 
//			(The express edition is available for free at: http://www.microsoft.com/express/2005/). 
//
//			This code is distributed under the terms of the GNU General Public Licence (see
//			below). 
//
//
///////////////////////////////////////////////////////////////

/*	This file is part of CUDAMCML.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/

#include <float.h> //for FLT_MAX 
#include <stdio.h>

#include <cuda_runtime.h>

#ifdef _WIN32 
  #include "cudamcml_io.c"
	#include "cutil-win32/cutil.h"
#else 
  #include <cutil.h>
#endif

#ifdef _WIN32 
	#include "cutil-win32/multithreading.h"
#else
	#include "multithreading.h"
#endif

#include "cudamcml.h"
#include "cudamcml_kernel.h"

#include "cudamcml_kernel.cu"
#include "cudamcml_mem.cu"


//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static CUT_THREADPROC RunGPUi(HostThreadState *hstate)
{
    SimState *HostMem = &(hstate->host_sim_state);
    SimState DeviceMem;
    GPUThreadStates tstates;

    cudaError_t cudastat;

	  CUDA_SAFE_CALL( cudaSetDevice(hstate->dev_id) );

    // Init the remaining states.
    InitSimStates(HostMem, &DeviceMem, &tstates, hstate->sim);
    CUDA_SAFE_CALL( cudaThreadSynchronize() ); // Wait for all threads to finish
    cudastat=cudaGetLastError(); // Check if there was an error
    if (cudastat)
    {
        fprintf(stderr, "[GPU %u] failure in InitSimStates (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(&DeviceMem, &tstates);
        exit(1);
    }

    InitDCMem(hstate->sim, hstate->A_rz_overflow);
    CUDA_SAFE_CALL( cudaThreadSynchronize() ); // Wait for all threads to finish
    cudastat=cudaGetLastError(); // Check if there was an error
    if (cudastat)
    {
        fprintf(stderr, "[GPU %u] failure in InitDCMem (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(&DeviceMem, &tstates);
        exit(1);
    }

    dim3 dimBlock(NUM_THREADS_PER_BLOCK);
    dim3 dimGrid(NUM_BLOCKS);

    // Initialize the remaining thread states.
    InitThreadState<<<dimGrid,dimBlock>>>(tstates);
    CUDA_SAFE_CALL( cudaThreadSynchronize() ); // Wait for all threads to finish
    cudastat=cudaGetLastError(); // Check if there was an error
    if (cudastat)
    {
        fprintf(stderr, "[GPU %u] failure in InitThreadState (%i): %s\n",
                hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
        FreeHostSimState(HostMem);
        FreeDeviceSimStates(&DeviceMem, &tstates);
        exit(1);
    }

    for (int i = 1; *HostMem->n_photons_left > 0; ++i)
    {
        // Run the kernel.
        if (hstate->sim->ignoreAdetection == 1)
        {
            MCMLKernel<1><<<dimGrid, dimBlock>>>(DeviceMem, tstates);
        }
        else
        {
            MCMLKernel<0><<<dimGrid, dimBlock>>>(DeviceMem, tstates);
        }
        CUDA_SAFE_CALL( cudaThreadSynchronize() ); // Wait for all threads to finish
        cudastat = cudaGetLastError();             // Check if there was an error
        if (cudastat)
        {
            fprintf(stderr, "[GPU %u] failure in MCMLKernel (%i): %s.\n",
                    hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
            FreeHostSimState(HostMem);
            FreeDeviceSimStates(&DeviceMem, &tstates);
            exit(1);
        }

        // Copy the number of photons left from device to host.
        CUDA_SAFE_CALL( cudaMemcpy(HostMem->n_photons_left,
                    DeviceMem.n_photons_left, sizeof(unsigned int),
                    cudaMemcpyDeviceToHost) );

        printf("[GPU %u] batch %d, number of photons left %u\n",
                hstate->dev_id, i, *(HostMem->n_photons_left));
    }

    printf("[GPU %u] simulation done!\n", hstate->dev_id);

    CopyDeviceToHostMem(HostMem, &DeviceMem, hstate->sim);
    FreeDeviceSimStates(&DeviceMem, &tstates);
    // We still need the host-side structure.
	
	//CUT_THREADEND;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

static void DoOneSimulation(int sim_id, SimulationStruct* simulation,
        unsigned int num_GPUs,
#ifdef USE_MT_RNG
        unsigned long long *x, unsigned int *a)
#else
        unsigned int *s1, unsigned int *s2, unsigned int *s3)
#endif
{
    printf("\n------------------------------------------------------------\n");
    printf("        Simulation #%d\n", sim_id);
    printf("        - number_of_photons = %u\n", simulation->number_of_photons);

    // Compute GPU-specific constant parameters.
    unsigned int A_rz_overflow = 0;
    // We only need it if we care about A_rz.
    if (! simulation->ignoreAdetection)
    {
        A_rz_overflow = compute_Arz_overflow_count(simulation->start_weight,
                simulation->layers, simulation->n_layers);
        printf("        - A_rz_overflow = %u\n", A_rz_overflow);
    }

    printf("------------------------------------------------------------\n\n");

    //Simulation kernel exec time
     unsigned int execTimer = 0;
     CUT_SAFE_CALL( cutCreateTimer( &execTimer));
     CUT_SAFE_CALL( cutStartTimer(execTimer));

//    clock_t time1,time2;
    // Start the clock
//    time1=clock();

    // Distribute all photons among GPUs.
    unsigned int n_photons_per_GPU = simulation->number_of_photons / num_GPUs;

    // For each GPU, init the host-side structure.
    HostThreadState * hstates [MAX_GPU_COUNT];
    for (unsigned int i = 0; i < num_GPUs; ++i)
    {
        hstates[i] = (HostThreadState*)malloc(sizeof(HostThreadState));

        hstates[i]->dev_id = i;
        hstates[i]->sim = simulation;
        hstates[i]->A_rz_overflow = A_rz_overflow;

        SimState *hss = &(hstates[i]->host_sim_state);

        // number of photons responsible 
        hss->n_photons_left = (unsigned int*)malloc(sizeof(unsigned int));
        // The last GPU may be responsible for more photons if the
        // distribution is uneven.
        *(hss->n_photons_left) = (i == num_GPUs-1) ?
            simulation->number_of_photons - (num_GPUs-1) * n_photons_per_GPU :
            n_photons_per_GPU;

        // random number seeds
        unsigned int ofst = i * NUM_THREADS;
#ifdef USE_MT_RNG
        hss->x = &x[ofst]; hss->a = &a[ofst];
#else
        hss->s1 = &s1[ofst]; hss->s2 = &s2[ofst]; hss->s3 = &s3[ofst];
#endif
    }

    // Launch a dedicated host thread for each GPU.
    CUTThread hthreads[MAX_GPU_COUNT];
    for (unsigned int i = 0; i < num_GPUs; ++i)
    {
        hthreads[i] = cutStartThread((CUT_THREADROUTINE)RunGPUi, hstates[i]);
    }

    // Wait for all host threads to finish.
    cutWaitForThreads(hthreads, num_GPUs);


    // Check any of the threads failed.
    int failed = 0;
    for (unsigned int i = 0; i < num_GPUs && !failed; ++i)
    {
        if (hstates[i]->host_sim_state.n_photons_left == NULL) failed = 1;
    }

    if (!failed)
    {
        // Sum the results to hstates[0].
        SimState *hss0 = &(hstates[0]->host_sim_state);
        for (unsigned int i = 1; i < num_GPUs; ++i)
        {
            SimState *hssi = &(hstates[i]->host_sim_state);

            // A_rz
            int size = simulation->det.nr * simulation->det.nz;
            for (int j = 0; j < size; ++j)
            {
                hss0->A_rz[j] += hssi->A_rz[j];
            }

            // Rd_ra
            size = simulation->det.na * simulation->det.nr;
            for (int j = 0; j < size; ++j)
            {
                hss0->Rd_ra[j] += hssi->Rd_ra[j];
            }

            // Tt_ra
            size = simulation->det.na * simulation->det.nr;
            for (int j = 0; j < size; ++j)
            {
                hss0->Tt_ra[j] += hssi->Tt_ra[j];
            }
        }


 //        time2=clock();
//        printf("\n*** Simulation time: %.3f sec\n",
//                (double)(time2-time1)/CLOCKS_PER_SEC);

        Write_Simulation_Results(hss0, simulation, (double)cutGetTimerValue(execTimer)/1000);
        
	CUT_SAFE_CALL( cutStopTimer(execTimer));
        printf( "\n\n>>>>>>Simulation time: %f (ms)\n", cutGetTimerValue(execTimer));
     	CUT_SAFE_CALL( cutDeleteTimer( execTimer));

    }

    // Free SimState structs.
    for (unsigned int i = 0; i < num_GPUs; ++i)
    {
        FreeHostSimState(&(hstates[i]->host_sim_state));
        free(hstates[i]);
    }
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
    int i;
    SimulationStruct* simulations;
    int n_simulations;
    unsigned long long seed = (unsigned long long) time(NULL);
    char* filename;
    int ignoreAdetection = 0;
    unsigned int num_GPUs = 1;

    if (argc < 2)
    {
        printf("Not enough input arguments!\n");
        print_usage();
        return 1;
    }
    filename = argv[1];

    // Parse command-line arguments.
    if (interpret_arg(argc, argv, &seed, &ignoreAdetection, &num_GPUs))
    {
        print_usage();
        return 1;
    }

    // Determine the number of GPUs available.
    int dev_count;
    CUDA_SAFE_CALL( cudaGetDeviceCount(&dev_count) );
    if (dev_count <= 0)
    {
        fprintf(stderr, "No GPU available! Quit.\n");
        return 1;
    }
    if (dev_count > MAX_GPU_COUNT) dev_count = MAX_GPU_COUNT;

    // Make sure we do not use more than what we have.
    if (num_GPUs > dev_count)
    {
        printf("The number of GPUs specified (%u) is more than "
                "what is available (%d)!\n", num_GPUs, dev_count);
        num_GPUs = (unsigned int)dev_count;
    }
    printf("Using %u GPUs ...\n", num_GPUs);

    // Read the simulation inputs.
    n_simulations = read_simulation_data(filename, &simulations,
            ignoreAdetection);
    if(n_simulations == 0)
    {
        printf("Something wrong with read_simulation_data!\n");
        return 1;
    }
    printf("Read %d simulations\n",n_simulations);

    // Allocate and initialize RNG seeds (for all threads on all GPUs).
    unsigned int len = NUM_THREADS * num_GPUs;
#ifdef USE_MT_RNG
    unsigned long long *x = (unsigned long long*)
        malloc(len * sizeof(unsigned long long));
    unsigned int *a = (unsigned int*)malloc(len * sizeof(unsigned int));
    if (init_RNG(x, a, len, "safeprimes_base32.txt", seed)) return 1;
    printf("Using the Mersenne Twister random number generator ...\n");
#else
    unsigned int *s1 = (unsigned int*)malloc(len * sizeof(unsigned int));
    unsigned int *s2 = (unsigned int*)malloc(len * sizeof(unsigned int));
    unsigned int *s3 = (unsigned int*)malloc(len * sizeof(unsigned int));
    init_Taus_seeds(s1, s2, s3, len);
    printf("Using the Tausworthe random number generator ...\n");
#endif

    //perform all the simulations
    for(i=0;i<n_simulations;i++)
    {
        // Run a simulation
#ifdef USE_MT_RNG
        DoOneSimulation(i, &simulations[i], num_GPUs, x, a);
#else
        DoOneSimulation(i, &simulations[i], num_GPUs, s1, s2, s3);
#endif
    }

    // Free the random number seed arrays.
#ifdef USE_MT_RNG
    free(x); free(a);
#else
    free(s1); free(s2); free(s3);
#endif

    FreeSimulationStruct(simulations, n_simulations);

    return 0; 
}
