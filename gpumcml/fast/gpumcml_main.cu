/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//   GPU-based Monte Carlo simulation of photon migration in multi-layered media (GPUMCML)
//   Copyright (C) 2009
//	
//	 Some documentation is available for GPUMCML and should have been distributed along 
//	 with this source code. If that is not the case: Documentation, source code and executables
//	 for GPUMCML are available for download on our webpage:
//   http://code.google.com/p/gpumcml 
// 
//	 http://www.atomic.physics.lu.se/Biophotonics
//	 or, directly
//	 http://www.atomic.physics.lu.se/fileadmin/atomfysik/Biophotonics/Software/CUDAMCML.zip
//
//	 We encourage the use, and modification of this code, and hope it will help 
//	 users/programmers to utilize the power of GPGPU for their simulation needs. While we
//	 don't have a scientifc publication describing this code, we would very much appreciate
//	 if you cite our original GPGPU Monte Carlo letter (on which GPUMCML is based) if you 
//	 use this code or derivations thereof for your own scientifc work:
//	 E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics processing
//	 units for high-speed Monte Carlo simulations of photon migration", Journal of Biomedical Optics
//	 Letters, 13(6) 060504 (2008).
//
//	 To compile and run this code, please visit www.nvidia.com and download the necessary 
//	 CUDA Toolkit and SKD. We also highly recommend the Visual Studio wizard 
//	 (available at:http://forums.nvidia.com/index.php?showtopic=69183) 
//	 if you use Visual Studio 2005 
//	 (The express edition is available for free at: http://www.microsoft.com/express/2005/). 
//  	
//   This code is distributed under the terms of the GNU General Public Licence (see below). 
//
/////////////////////////////////////////////////////////////////////////////////////////////////////////
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


#include <float.h> //for FLT_MAX 
#include <stdio.h>

#include <cuda_runtime.h>

#ifdef _WIN32 
#include "gpumcml_io.c"
#include "cutil-win32/cutil.h"
#else 
#include <cutil.h>
#endif

#ifdef _WIN32 
#include "cutil-win32/multithreading.h"
#else
#include "multithreading.h"
#endif

#include "gpumcml.h"
#include "gpumcml_kernel.h"

#include "gpumcml_kernel.cu"
#include "gpumcml_mem.cu"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//   Supports multiple GPUs by allowing multiple host threads to launch kernel
//   Each thread calls RunGPUi with its own HostThreadState parameters
//////////////////////////////////////////////////////////////////////////////
static CUT_THREADPROC RunGPUi(HostThreadState *hstate)
{
  SimState *HostMem = &(hstate->host_sim_state);
  SimState DeviceMem;
  GPUThreadStates tstates;
  // total number of threads in the grid
  UINT32 n_threads = hstate->n_tblks * NUM_THREADS_PER_BLOCK;
  cudaError_t cudastat;

  CUDA_SAFE_CALL( cudaSetDevice(hstate->dev_id) );

  // Init the remaining states.
  InitSimStates(HostMem, &DeviceMem, &tstates, hstate->sim, n_threads);
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
  dim3 dimGrid(hstate->n_tblks);

  int k_smem_sz = 0;
#ifdef USE_32B_ELEM_FOR_ARZ_SMEM
  // This piece of shared memory is for overflow handling.
  k_smem_sz = NUM_THREADS_PER_BLOCK * sizeof(UINT32);
#endif

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

#ifdef USE_TRUE_CACHE
  // Configure the L1 cache for Fermi.
  if (hstate->sim->ignoreAdetection == 1)
  {
    cudaFuncSetCacheConfig(MCMLKernel<1>, cudaFuncCachePreferL1);
  }
  else
  {
    cudaFuncSetCacheConfig(MCMLKernel<0>, cudaFuncCachePreferL1);
  }
#endif

  for (int i = 1; *HostMem->n_photons_left > 0; ++i)
  {
    // Run the kernel.
    if (hstate->sim->ignoreAdetection == 1)
    {
      MCMLKernel<1><<<dimGrid, dimBlock, k_smem_sz>>>(DeviceMem, tstates);
    }
    else
    {
      MCMLKernel<0><<<dimGrid, dimBlock, k_smem_sz>>>(DeviceMem, tstates);
    }
    // Wait for all threads to finish.
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    // Check if there was an error
    cudastat = cudaGetLastError();
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

    printf("[GPU %u] batch %5d, number of photons left %10u\n",
      hstate->dev_id, i, *(HostMem->n_photons_left));
  }

  // Sum the multiple copies of A_rz in the global memory.
  sum_A_rz<<<30, 128>>>(DeviceMem.A_rz);
  // Wait for all threads to finish.
  CUDA_SAFE_CALL( cudaThreadSynchronize() );
  // Check if there was an error
  cudastat = cudaGetLastError();
  if (cudastat)
  {
    fprintf(stderr, "[GPU %u] failure in sum_A_rz (%i): %s.\n",
        hstate->dev_id, cudastat, cudaGetErrorString(cudastat));
    FreeHostSimState(HostMem);
    FreeDeviceSimStates(&DeviceMem, &tstates);
    exit(1); 
  }

  printf("[GPU %u] simulation done!\n", hstate->dev_id);

  CopyDeviceToHostMem(HostMem, &DeviceMem, hstate->sim, n_threads);
  FreeDeviceSimStates(&DeviceMem, &tstates);
  // We still need the host-side structure.
}

//////////////////////////////////////////////////////////////////////////////
//   Perform MCML simulation for one run out of N runs (in the input file)
//////////////////////////////////////////////////////////////////////////////
static void DoOneSimulation(int sim_id, SimulationStruct* simulation,
                            HostThreadState* hstates[], UINT32 num_GPUs,
                            UINT64 *x, UINT32 *a)
{
  printf("\n------------------------------------------------------------\n");
  printf("        Simulation #%d\n", sim_id);
  printf("        - number_of_photons = %u\n", simulation->number_of_photons);

  // Compute GPU-specific constant parameters.
  UINT32 A_rz_overflow = 0;
  // We only need it if we care about A_rz.
#if !defined(USE_TRUE_CACHE) && defined(USE_32B_ELEM_FOR_ARZ_SMEM)
  if (! simulation->ignoreAdetection)
  {
    A_rz_overflow = compute_Arz_overflow_count(simulation->start_weight,
        simulation->layers, simulation->n_layers, NUM_THREADS_PER_BLOCK);
    printf("        - A_rz_overflow = %u\n", A_rz_overflow);
  }
#endif

  printf("------------------------------------------------------------\n\n");

  cudaEvent_t start, stop;
  float elapsedTime;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Start the timer.
  cudaEventRecord(start,0);

  // Distribute all photons among GPUs.
  unsigned int n_photons_per_GPU = simulation->number_of_photons / num_GPUs;

  // For each GPU, init the host-side structure.
  for (UINT32 i = 0; i < num_GPUs; ++i)
  {
    hstates[i]->sim = simulation;
    hstates[i]->A_rz_overflow = A_rz_overflow;

    SimState *hss = &(hstates[i]->host_sim_state);

    // number of photons responsible 
    hss->n_photons_left = (UINT32*)malloc(sizeof(UINT32));
    // The last GPU may be responsible for more photons if the
    // distribution is uneven.
    *(hss->n_photons_left) = (i == num_GPUs-1) ?
      simulation->number_of_photons - (num_GPUs-1) * n_photons_per_GPU :
    n_photons_per_GPU;
  }

  // Launch a dedicated host thread for each GPU.
  CUTThread hthreads[MAX_GPU_COUNT];
  for (UINT32 i = 0; i < num_GPUs; ++i)
  {
    hthreads[i] = cutStartThread((CUT_THREADROUTINE)RunGPUi, hstates[i]);
  }

  // Wait for all host threads to finish.
  cutWaitForThreads(hthreads, num_GPUs);


  // Check any of the threads failed.
  int failed = 0;
  for (UINT32 i = 0; i < num_GPUs && !failed; ++i)
  {
    if (hstates[i]->host_sim_state.n_photons_left == NULL) failed = 1;
  }

  if (!failed)
  {
    // Sum the results to hstates[0].
    SimState *hss0 = &(hstates[0]->host_sim_state);
    for (UINT32 i = 1; i < num_GPUs; ++i)
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

    // End the timer.
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    // Compute the execution time.
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // Convert to seconds.
    elapsedTime /= 1000.0;
    printf("\n*** Simulation time: %.3f sec\n\n", elapsedTime);

    Write_Simulation_Results(hss0, simulation, elapsedTime);
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Free SimState structs.
  for (UINT32 i = 0; i < num_GPUs; ++i)
  {
    FreeHostSimState(&(hstates[i]->host_sim_state));
  }
}

//////////////////////////////////////////////////////////////////////////////
//   Perform MCML simulation for one run out of N runs (in the input file)
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
  char* filename = NULL;
  UINT64 seed = (UINT64) time(NULL);
  int ignoreAdetection = 0;
  UINT32 num_GPUs = 1;

  SimulationStruct* simulations;
  int n_simulations;

  int i;

  // Parse command-line arguments.
  if (interpret_arg(argc, argv, &filename,
    &seed, &ignoreAdetection, &num_GPUs))
  {
    usage(argv[0]);
    return 1;
  }

  // Determine the number of GPUs available.
  int dev_count;
  CUDA_SAFE_CALL( cudaGetDeviceCount(&dev_count) );
  if (dev_count <= 0)
  {
    fprintf(stderr, "No GPU available. Quit.\n");
    return 1;
  }

  // Make sure we do not use more than what we have.
  if (num_GPUs > dev_count)
  {
    printf("The number of GPUs specified (%u) is more than "
      "what is available (%d)!\n", num_GPUs, dev_count);
    num_GPUs = (UINT32)dev_count;
  }

  // Output the execution configuration.
  printf("\n====================================\n");
  printf("EXECUTION MODE:\n");
  printf("  ignore A-detection:      %s\n",
    ignoreAdetection ? "YES" : "NO");
  printf("  seed:                    %llu\n", seed);
  printf("  # of GPUs:               %u\n", num_GPUs);
  printf("====================================\n\n");

  // Read the simulation inputs.
  n_simulations = read_simulation_data(filename, &simulations,
    ignoreAdetection);
  if(n_simulations == 0)
  {
    printf("Something wrong with read_simulation_data!\n");
    return 1;
  }
  printf("Read %d simulations\n\n",n_simulations);

  // Allocate one host thread state for each GPU.
  HostThreadState* hstates[MAX_GPU_COUNT];
  cudaDeviceProp props;
  int n_threads = 0;    // total number of threads for all GPUs
  for (i = 0; i < num_GPUs; ++i)
  {
    hstates[i] = (HostThreadState*)malloc(sizeof(HostThreadState));

    // Set the GPU ID.
    hstates[i]->dev_id = i;

    // Get the GPU properties.
    CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, hstates[i]->dev_id) );
    printf("[GPU %u] \"%s\" with Compute Capability %d.%d (%d SMs)\n",
        i, props.name, props.major, props.minor, props.multiProcessorCount);

    // Validate the GPU compute capability.
    int cc = props.major * 10 + props.minor;
    if (cc < CUDA_ARCH)
    {
      fprintf(stderr, "\nGPU %u does not meet the Compute Capability "
          "this program requires (%d)! Abort.\n\n", i, CUDA_ARCH);
      exit(1);
    }

    // We launch one thread block for each SM on this GPU.
    hstates[i]->n_tblks = props.multiProcessorCount;

    n_threads += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
  }

  // Allocate and initialize RNG seeds (for all threads on all GPUs).
  UINT64 *x = (UINT64*)malloc(n_threads * sizeof(UINT64));
  UINT32 *a = (UINT32*)malloc(n_threads * sizeof(UINT32));
  if (init_RNG(x, a, n_threads, "safeprimes_base32.txt", seed)) return 1;
  printf("\nUsing the MWC random number generator ...\n");

  // Assign these seeds to each host thread state.
  int ofst = 0;
  for (i = 0; i < num_GPUs; ++i)
  {
    SimState *hss = &(hstates[i]->host_sim_state);
    hss->x = &x[ofst];
    hss->a = &a[ofst];

    ofst += hstates[i]->n_tblks * NUM_THREADS_PER_BLOCK;
  }

  //perform all the simulations
  for(i=0;i<n_simulations;i++)
  {
    // Run a simulation
    DoOneSimulation(i, &simulations[i], hstates, num_GPUs, x, a);
  }

  // Free host thread states.
  for (i = 0; i < num_GPUs; ++i) free(hstates[i]);

  // Free the random number seed arrays.
  free(x); free(a);

  FreeSimulationStruct(simulations, n_simulations);

  return 0; 
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

