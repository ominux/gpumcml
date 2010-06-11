/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//   GPU-based Monte Carlo simulation of photon migration in multi-layered media (GPU-MCML)
//   Copyright (C) 2009
//	
//   || DEVELOPMENT TEAM: 
//   --------------------------------------------------------------------------------------------------
//   Erik Alerstam, David Han, and William C. Y. Lo
//   
//   This code is the result of the collaborative efforts between 
//   Lund University and the University of Toronto.  
//
//   || DOCUMENTATION AND USER MANUAL: 
//   --------------------------------------------------------------------------------------------------
//	 Detailed "Wiki" style documentation is being developed for GPU-MCML 
//   and will be available on our webpage soon:
//   http://code.google.com/p/gpumcml 
// 
//   || NEW FEATURES: 
//   --------------------------------------------------------------------------------------------------
//    - Supports the Fermi GPU architecture 
//    - Multi-GPU execution 
//    - Automatic selection of optimization parameters  
//    - Backward compatible on pre-Fermi graphics cards
//    - Supports linux and Windows environment (Visual Studio)
//   
//   || PREVIOUS WORK: 
//   --------------------------------------------------------------------------------------------------
//	 This code is the fusion of our earlier, preliminary implementations and combines the best features 
//   from each implementation.  
//
//   W. C. Y. Lo, T. D. Han, J. Rose, and L. Lilge, "GPU-accelerated Monte Carlo simulation for photodynamic
//   therapy treatment planning," in Proc. of SPIE-OSA Biomedical Optics, vol. 7373.
//   
//   and 
//
//   http://www.atomic.physics.lu.se/biophotonics/our_research/monte_carlo_simulations/gpu_monte_carlo/
//	 E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics processing
//	 units for high-speed Monte Carlo simulations of photon migration", Journal of Biomedical Optics
//	 Letters, 13(6) 060504 (2008).
//
//   || CITATION: 
//   --------------------------------------------------------------------------------------------------
//	 We encourage the use, and modification of this code, and hope it will help 
//	 users/programmers to utilize the power of GPGPU for their simulation needs. While we
//	 don't have a scientific publication describing this code yet, we would very much appreciate it
//	 if you cite our original papers above if you use this code or derivations 
//   thereof for your own scientific work
//
//	 To compile and run this code, please visit www.nvidia.com and download the necessary 
//	 CUDA Toolkit, SDK, and Developer Drivers 
//
//	 If you use Visual Studio, the express edition is available for free at 
//   http://www.microsoft.com/express/Downloads/). 
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

#include "gpumcml.h"
#include "gpumcml_kernel.h"

#include "gpumcml_kernel.cu"
#include "gpumcml_mem.cu"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//   Supports 1 GPU only
//   Calls RunGPU with HostThreadState parameters
//////////////////////////////////////////////////////////////////////////////
static void RunGPUi(HostThreadState *hstate)
{
  SimState *HostMem = &(hstate->host_sim_state);
  SimState DeviceMem;
  GPUThreadStates tstates;

  cudaError_t cudastat;

  // Init the remaining states.
  InitSimStates(HostMem, &DeviceMem, &tstates, hstate->sim);

  InitDCMem(hstate->sim);

  dim3 dimBlock(NUM_THREADS_PER_BLOCK);
  dim3 dimGrid(NUM_BLOCKS);

  // Initialize the remaining thread states.
  InitThreadState<<<dimGrid,dimBlock>>>(tstates);

  // Configure the L1 cache for Fermi.
#ifdef USE_TRUE_CACHE
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
      MCMLKernel<1><<<dimGrid, dimBlock>>>(DeviceMem, tstates);
    }
    else
    {
      MCMLKernel<0><<<dimGrid, dimBlock>>>(DeviceMem, tstates);
    }

    // Check if there was an error
    cudastat = cudaGetLastError();
    if (cudastat)
    {
      fprintf(stderr, "[GPU] failure in MCMLKernel (%i): %s.\n",cudastat, cudaGetErrorString(cudastat));
      FreeHostSimState(HostMem);
      FreeDeviceSimStates(&DeviceMem, &tstates);
      exit(1); 
    }

    // Copy the number of photons left from device to host.
    CUDA_SAFE_CALL( cudaMemcpy(HostMem->n_photons_left,
      DeviceMem.n_photons_left, sizeof(unsigned int),
      cudaMemcpyDeviceToHost) );

    printf("[GPU] batch %5d, number of photons left %10u\n",i, *(HostMem->n_photons_left));
  }

  printf("[GPU] simulation done!\n");

  CopyDeviceToHostMem(HostMem, &DeviceMem, hstate->sim);
  FreeDeviceSimStates(&DeviceMem, &tstates);
  // We still need the host-side structure.
}

//////////////////////////////////////////////////////////////////////////////
//   Perform MCML simulation for one run out of N runs (in the input file)
//////////////////////////////////////////////////////////////////////////////
static void DoOneSimulation(int sim_id, SimulationStruct* simulation,
                            unsigned long long *x, unsigned int *a)
{
  printf("\n------------------------------------------------------------\n");
  printf("        Simulation #%d\n", sim_id);
  printf("        - number_of_photons = %u\n", simulation->number_of_photons);
  printf("------------------------------------------------------------\n\n");

  // Start simulation kernel exec timer
  unsigned int execTimer = 0;
  CUT_SAFE_CALL( cutCreateTimer( &execTimer));
  CUT_SAFE_CALL( cutStartTimer(execTimer));

  //clock_t time1,time2;

  //// Start the clock
  //time1=clock();

  // For each GPU, init the host-side structure.
  HostThreadState* hstates;
  hstates = (HostThreadState*)malloc(sizeof(HostThreadState));

  hstates->sim = simulation;

  SimState *hss = &(hstates->host_sim_state);

  // number of photons responsible 
  hss->n_photons_left = (unsigned int*)malloc(sizeof(unsigned int));
  *(hss->n_photons_left) = simulation->number_of_photons; 

  // random number seeds
  hss->x = &x[0]; hss->a = &a[0];

  // Launch simulation
  RunGPUi (hstates);

  // End the timer.
  //time2=clock();
  //printf("\n*** Simulation time: %.3f sec\n\n",(double)(time2-time1)/CLOCKS_PER_SEC);
  CUT_SAFE_CALL( cutStopTimer(execTimer));
  printf( "\n\n>>>>>>Simulation time: %f (ms)\n", cutGetTimerValue(execTimer));
  
  Write_Simulation_Results(hss, simulation, cutGetTimerValue(execTimer));

  // Free SimState structs.
  FreeHostSimState(hss);
  free(hstates);
  CUT_SAFE_CALL( cutDeleteTimer( execTimer));
}

//////////////////////////////////////////////////////////////////////////////
//   Perform MCML simulation for one run out of N runs (in the input file)
//////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	//Init GPU Device
	CUT_DEVICE_INIT(argc, argv);

  char* filename = NULL;
  unsigned long long seed = (unsigned long long) time(NULL);
  int ignoreAdetection = 0;
  
  SimulationStruct* simulations;
  int n_simulations;

  int i;

  // Parse command-line arguments.
  if (interpret_arg(argc, argv, &filename,&seed, &ignoreAdetection))
  {
    usage(argv[0]);
    return 1;
  }

  // Output the execution configuration.
  printf("\n====================================\n");
  printf("EXECUTION MODE:\n");
  printf("  ignore A-detection:      %s\n", ignoreAdetection ? "YES" : "NO");
  printf("  seed:                    %llu\n", seed);
  printf("====================================\n\n");

  // Read the simulation inputs.
  n_simulations = read_simulation_data(filename, &simulations, ignoreAdetection);
  if(n_simulations == 0)
  {
    printf("Something wrong with read_simulation_data!\n");
    return 1;
  }
  printf("Read %d simulations\n",n_simulations);

  // Allocate and initialize RNG seeds.
  unsigned int len = NUM_THREADS;

  unsigned long long *x = (unsigned long long*)malloc(len * sizeof(unsigned long long));
  unsigned int *a = (unsigned int*)malloc(len * sizeof(unsigned int));

#ifdef _WIN32 
  if (init_RNG(x, a, len, "safeprimes_base32.txt", seed)) return 1;
#else 
  if (init_RNG(x, a, len, "executable/safeprimes_base32.txt", seed)) return 1;
#endif
  
  printf("Using the MWC random number generator ...\n");

  //perform all the simulations
  for(i=0;i<n_simulations;i++)
  {
    // Run a simulation
    DoOneSimulation(i, &simulations[i], x, a);
  }

  // Free the random number seed arrays.
  free(x); free(a);
  FreeSimulationStruct(simulations, n_simulations);

  return 0; 
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////