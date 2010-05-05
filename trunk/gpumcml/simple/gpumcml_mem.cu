/*****************************************************************************
*
*   GPU memory allocation, initialization, and transfer (Host <--> GPU)
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

#include <stdio.h>

#include "gpumcml_kernel.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//   Initialize Device Constant Memory with read-only data
//////////////////////////////////////////////////////////////////////////////
int InitDCMem(SimulationStruct *sim)
{
  // Make sure that the number of layers is within the limit.
  UINT32 n_layers = sim->n_layers + 2;
  if (n_layers > MAX_LAYERS) return 1;

  SimParamGPU h_simparam;

  h_simparam.num_layers = sim->n_layers;  // not plus 2 here
  h_simparam.init_photon_w = sim->start_weight;
  h_simparam.dz = sim->det.dz;
  h_simparam.dr = sim->det.dr;
  h_simparam.na = sim->det.na;
  h_simparam.nz = sim->det.nz;
  h_simparam.nr = sim->det.nr;

  CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_simparam,
    &h_simparam, sizeof(SimParamGPU)) );

  LayerStructGPU h_layerspecs[MAX_LAYERS];

  for (UINT32 i = 0; i < n_layers; ++i)
  {
    h_layerspecs[i].z0 = sim->layers[i].z_min;
    h_layerspecs[i].z1 = sim->layers[i].z_max;
    FLOAT n1 = sim->layers[i].n;
    h_layerspecs[i].n = n1;

    // TODO: sim->layer should not do any pre-computation.
    FLOAT rmuas = sim->layers[i].mutr;
    h_layerspecs[i].muas = FP_ONE / rmuas;
    h_layerspecs[i].rmuas = rmuas;
    h_layerspecs[i].mua_muas = sim->layers[i].mua * rmuas;

    h_layerspecs[i].g = sim->layers[i].g;

    if (i == 0 || i == n_layers-1)
    {
      h_layerspecs[i].cos_crit0 = MCML_FP_ZERO;
      h_layerspecs[i].cos_crit1 = MCML_FP_ZERO;
    }
    else
    {
      FLOAT n2 = sim->layers[i-1].n;
      h_layerspecs[i].cos_crit0 = (n1 > n2) ?
        sqrtf(FP_ONE - n2*n2/(n1*n1)) : MCML_FP_ZERO;
      n2 = sim->layers[i+1].n;
      h_layerspecs[i].cos_crit1 = (n1 > n2) ?
        sqrtf(FP_ONE - n2*n2/(n1*n1)) : MCML_FP_ZERO;
    }
  }

  // Copy layer data to constant device memory
  CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_layerspecs,
    &h_layerspecs, n_layers*sizeof(LayerStructGPU)) );

  return 0;
}

//////////////////////////////////////////////////////////////////////////////
//   Initialize Device Memory (global) for read/write data
//////////////////////////////////////////////////////////////////////////////
// DAVID
int InitSimStates(SimState* HostMem, SimState* DeviceMem,
                  GPUThreadStates *tstates, SimulationStruct* sim)
{
  int rz_size = sim->det.nr * sim->det.nz;
  int ra_size = sim->det.nr * sim->det.na;

  unsigned int size;

  // Allocate n_photons_left (on device only)
  size = sizeof(UINT32);
  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->n_photons_left, size) );
  CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->n_photons_left,
    HostMem->n_photons_left, size, cudaMemcpyHostToDevice) );

  // random number generation (on device only)
  size = NUM_THREADS * sizeof(UINT32);

  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->a, size) );
  CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->a, HostMem->a, size,
    cudaMemcpyHostToDevice) );
  size = NUM_THREADS * sizeof(UINT64);
  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->x, size) );
  CUDA_SAFE_CALL( cudaMemcpy(DeviceMem->x, HostMem->x, size,
    cudaMemcpyHostToDevice) );


  // Allocate A_rz on host and device
  size = rz_size * sizeof(UINT64);
  HostMem->A_rz = (UINT64*)malloc(size);
  if (HostMem->A_rz == NULL)
  {
    fprintf(stderr, "Error allocating HostMem->A_rz");
    exit(1);
  }
  // On the device, we allocate multiple copies for less access contention.
  //size *= N_A_RZ_COPIES;
  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->A_rz, size) );
  CUDA_SAFE_CALL( cudaMemset(DeviceMem->A_rz, 0, size) );

  // Allocate Rd_ra on host and device
  size = ra_size * sizeof(UINT64);
  HostMem->Rd_ra = (UINT64*)malloc(size);
  if(HostMem->Rd_ra==NULL){printf("Error allocating HostMem->Rd_ra"); exit (1);}
  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Rd_ra, size) );
  CUDA_SAFE_CALL( cudaMemset(DeviceMem->Rd_ra, 0, size) );

  // Allocate Tt_ra on host and device
  size = ra_size * sizeof(UINT64);
  HostMem->Tt_ra = (UINT64*)malloc(size);
  if(HostMem->Tt_ra==NULL){printf("Error allocating HostMem->Tt_ra"); exit (1);}
  CUDA_SAFE_CALL( cudaMalloc((void**)&DeviceMem->Tt_ra, size) );
  CUDA_SAFE_CALL( cudaMemset(DeviceMem->Tt_ra, 0, size) );

  /* Allocate and initialize GPU thread states on the device.
  *
  * We only initialize rnd_a and rnd_x here. For all other fields, whose
  * initial value is a known constant, we use a kernel to do the
  * initialization.
  */

  // photon structure
  size = NUM_THREADS * sizeof(FLOAT);
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_x, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_y, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_z, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_ux, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_uy, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_uz, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_w, size) );
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_sleft, size) );
  size = NUM_THREADS * sizeof(UINT32);
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->photon_layer, size) );

  // thread active
  CUDA_SAFE_CALL( cudaMalloc((void**)&tstates->is_active, size) );

  return 1;
}

//////////////////////////////////////////////////////////////////////////////
//   Transfer data from Device to Host memory after simulation
//////////////////////////////////////////////////////////////////////////////
int CopyDeviceToHostMem(SimState* HostMem, SimState* DeviceMem, SimulationStruct* sim)
{
  int rz_size = sim->det.nr*sim->det.nz;
  int ra_size = sim->det.nr*sim->det.na;

  // Copy A_rz, Rd_ra and Tt_ra
  CUDA_SAFE_CALL( cudaMemcpy(HostMem->A_rz,DeviceMem->A_rz,rz_size*sizeof(UINT64),cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy(HostMem->Rd_ra,DeviceMem->Rd_ra,ra_size*sizeof(UINT64),cudaMemcpyDeviceToHost) );
  CUDA_SAFE_CALL( cudaMemcpy(HostMem->Tt_ra,DeviceMem->Tt_ra,ra_size*sizeof(UINT64),cudaMemcpyDeviceToHost) );

  //Also copy the state of the RNG's
  CUDA_SAFE_CALL( cudaMemcpy(HostMem->x,DeviceMem->x,NUM_THREADS*sizeof(UINT64),cudaMemcpyDeviceToHost) );

  return 0;
}

//////////////////////////////////////////////////////////////////////////////
//   Free Host Memory
//////////////////////////////////////////////////////////////////////////////
void FreeHostSimState(SimState *hstate)
{
  if (hstate->n_photons_left != NULL)
  {
    free(hstate->n_photons_left); hstate->n_photons_left = NULL;
  }

  // DO NOT FREE RANDOM NUMBER SEEDS HERE.

  if (hstate->A_rz != NULL)
  {
    free(hstate->A_rz); hstate->A_rz = NULL;
  }
  if (hstate->Rd_ra != NULL)
  {
    free(hstate->Rd_ra); hstate->Rd_ra = NULL;
  }
  if (hstate->Tt_ra != NULL)
  {
    free(hstate->Tt_ra); hstate->Tt_ra = NULL;
  }
}

//////////////////////////////////////////////////////////////////////////////
//   Free GPU Memory
//////////////////////////////////////////////////////////////////////////////
void FreeDeviceSimStates(SimState *dstate, GPUThreadStates *tstates)
{
  cudaFree(dstate->n_photons_left); dstate->n_photons_left = NULL;

  cudaFree(dstate->x); dstate->x = NULL;
  cudaFree(dstate->a); dstate->a = NULL;

  cudaFree(dstate->A_rz); dstate->A_rz = NULL;
  cudaFree(dstate->Rd_ra); dstate->Rd_ra = NULL;
  cudaFree(dstate->Tt_ra); dstate->Tt_ra = NULL;

  cudaFree(tstates->photon_x); tstates->photon_x = NULL;
  cudaFree(tstates->photon_y); tstates->photon_y = NULL;
  cudaFree(tstates->photon_z); tstates->photon_z = NULL;
  cudaFree(tstates->photon_ux); tstates->photon_ux = NULL;
  cudaFree(tstates->photon_uy); tstates->photon_uy = NULL;
  cudaFree(tstates->photon_uz); tstates->photon_uz = NULL;
  cudaFree(tstates->photon_w); tstates->photon_w = NULL;
  cudaFree(tstates->photon_sleft); tstates->photon_sleft = NULL;
  cudaFree(tstates->photon_layer); tstates->photon_layer = NULL;
  cudaFree(tstates->is_active); tstates->is_active = NULL;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////