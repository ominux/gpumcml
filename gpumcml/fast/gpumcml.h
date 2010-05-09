/*****************************************************************************
 *
 *   Header file for common data structures and constants (CPU and GPU) 
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


#ifndef _GPUMCML_H_
#define _GPUMCML_H_

#define SINGLE_PRECISION

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Various data types
typedef unsigned long long UINT64;
typedef unsigned int UINT32;

// MCML constants
#ifdef SINGLE_PRECISION
typedef float FLOAT;

// Critical weight for roulette
#define WEIGHT 1E-4F        

// scaling factor for photon weight, which is then converted to integer
#define WEIGHT_SCALE 16777216

#define PI_const 3.1415926F
#define RPI 0.318309886F

//NOTE: Single Precision
#define COSNINETYDEG 1.0E-6F
#define COSZERO (1.0F - 1.0E-6F)   
#define CHANCE 0.1F

#define MCML_FP_ZERO 0.0F
#define FP_ONE  1.0F
#define FP_TWO  2.0F

#else
typedef double FLOAT;

// Critical weight for roulette
#define WEIGHT 1E-4     

// scaling factor for photon weight, which is then converted to integer
#define WEIGHT_SCALE 12000000

#define PI_const 3.1415926
#define RPI 0.318309886

//NOTE: Double Precision
#define COSNINETYDEG 1.0E-6
#define COSZERO (1.0 - 1.0E-12)    
#define CHANCE 0.1

#define MCML_FP_ZERO 0.0
#define FP_ONE  1.0
#define FP_TWO  2.0

#endif

#define STR_LEN 200

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Data structure for specifying each layer
typedef struct
{
  FLOAT z_min;		// Layer z_min [cm]
  FLOAT z_max;		// Layer z_max [cm]
  FLOAT mutr;			// Reciprocal mu_total [cm]
  FLOAT mua;			// Absorption coefficient [1/cm]
  FLOAT g;			  // Anisotropy factor [-]
  FLOAT n;			  // Refractive index [-]
} LayerStruct;

// Detection Grid specifications
typedef struct
{
  FLOAT dr;		    // Detection grid resolution, r-direction [cm]
  FLOAT dz;		    // Detection grid resolution, z-direction [cm]

  UINT32 na;		  // Number of grid elements in angular-direction [-]
  UINT32 nr;		  // Number of grid elements in r-direction
  UINT32 nz;		  // Number of grid elements in z-direction
} DetStruct;

// Simulation input parameters 
typedef struct 
{
  char outp_filename[STR_LEN];
  char inp_filename[STR_LEN];

  // the starting and ending offset (in the input file) for this simulation
  long begin, end;
  // ASCII or binary output
  char AorB;

  UINT32 number_of_photons;
  int ignoreAdetection;
  FLOAT start_weight;

  DetStruct det;

  UINT32 n_layers;
  LayerStruct* layers;
} SimulationStruct;

// Per-GPU simulation states
// One instance of this struct exists in the host memory, while the other
// in the global memory.
typedef struct
{
  // points to a scalar that stores the number of photons that are not
  // completed (i.e. either on the fly or not yet started)
  UINT32 *n_photons_left;

  // per-thread seeds for random number generation
  // arrays of length NUM_THREADS
  // We put these arrays here as opposed to in GPUThreadStates because
  // they live across different simulation runs and must be copied back
  // to the host.
  UINT64 *x;
  UINT32 *a;

  // output data
  UINT64* Rd_ra;
  UINT64* A_rz;			// Pointer to a 2D absorption matrix!
  UINT64* Tt_ra;
} SimState;

// Everything a host thread needs to know in order to run simulation on
// one GPU (host-side only)
typedef struct
{
  // GPU identifier
  unsigned int dev_id;        

  // those states that will be updated
  SimState host_sim_state;

  // simulation input parameters
  SimulationStruct *sim;

  /* GPU-specific constant parameters */

  // number of thread blocks launched
  UINT32 n_tblks;

  // the limit that indicates overflow of an element of A_rz
  // in the shared memory
  UINT32 A_rz_overflow;

} HostThreadState;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern void usage(const char *prog_name);

// Parse the command-line arguments.
// Return 0 if successfull or a +ive error code.
extern int interpret_arg(int argc, char* argv[], char **fpath_p,
        unsigned long long* seed,
        int* ignoreAdetection, unsigned int *num_GPUs);

extern int read_simulation_data(char* filename,
        SimulationStruct** simulations, int ignoreAdetection);

extern int Write_Simulation_Results(SimState* HostMem,
        SimulationStruct* sim, float simulation_time);

extern void FreeSimulationStruct(SimulationStruct* sim, int n_simulations);

#endif  // _GPUMCML_H_
