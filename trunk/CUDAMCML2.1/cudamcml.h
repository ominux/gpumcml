#ifndef _CUDAMCML_H_
#define _CUDAMCML_H_

// Use the Mersenne Twister random number generator
// (as opposed to Tausworthe generator)
// #define USE_MT_RNG

#define SINGLE_PRECISION

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// Various data types
typedef float FLOAT;
typedef unsigned long long UINT64;
typedef unsigned int UINT32;

// MCML constants
#ifdef SINGLE_PRECISION

#define WEIGHT 1E-4F        /* Critical weight for roulette. */

// scaling factor for photon weight, which is then converted to integer
#define WEIGHT_SCALE 12000000

#define PI_const 3.1415926F
#define RPI 0.318309886F

//NOTE: ********#define COSZERO (1.0-1.0E-12) infers double data type
#define COSNINETYDEG 1.0E-6F
#define COSZERO (1.0F - 1.0E-6F)    // NOTE: 1.0-1.0E-7 doesn't work in CUDA!
#define CHANCE 0.1F

#define MCML_FP_ZERO 0.0F
#define FP_ONE  1.0F
#define FP_TWO  2.0F

#else

#define WEIGHT 1E-4     /* Critical weight for roulette. */

// scaling factor for photon weight, which is then converted to integer
#define WEIGHT_SCALE 12000000

#define PI_const 3.1415926
#define RPI 0.318309886

//NOTE: ********#define COSZERO (1.0-1.0E-12) infers double data type
#define COSNINETYDEG 1.0E-6
#define COSZERO (1.0 - 1.0E-6)    // NOTE: 1.0-1.0E-7 doesn't work in CUDA!
#define CHANCE 0.1

#define MCML_FP_ZERO 0.0
#define FP_ONE  1.0
#define FP_TWO  2.0

#endif

#define STR_LEN 200

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct
{
    FLOAT z_min;		// Layer z_min [cm]
    FLOAT z_max;		// Layer z_max [cm]
    FLOAT mutr;			// Reciprocal mu_total [cm]
    FLOAT mua;			// Absorption coefficient [1/cm]
    FLOAT g;			// Anisotropy factor [-]
    FLOAT n;			// Refractive index [-]
} LayerStruct;

typedef struct
{
    FLOAT dr;		// Detection grid resolution, r-direction [cm]
    FLOAT dz;		// Detection grid resolution, z-direction [cm]

    UINT32 na;		// Number of grid elements in angular-direction [-]
    UINT32 nr;		// Number of grid elements in r-direction
    UINT32 nz;		// Number of grid elements in z-direction
} DetStruct;

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
#ifdef USE_MT_RNG
    UINT64 *x;
    UINT32 *a;
#else
    UINT32 *s1;
    UINT32 *s2;
    UINT32 *s3;
#endif

    // output data
    UINT64* Rd_ra;
    UINT64* A_rz;			// Pointer to the 2D detection matrix!
    UINT64* Tt_ra;
} SimState;

// Everything a host thread needs to know in order to run simulation on
// one GPU (host-side only)
typedef struct
{
    unsigned int dev_id;        // GPU identifier

    // those states that will be updated
    SimState host_sim_state;

    // constant input parameters
    SimulationStruct *sim;

} HostThreadState;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern void usage(const char *prog_name);

// Parse the command-line arguments.
// Return 0 if successfull or a +ive error code.
//
extern int interpret_arg(int argc, char* argv[], char **fpath_p,
        unsigned long long* seed,
        int* ignoreAdetection, unsigned int *num_GPUs);

extern int read_simulation_data(char* filename,
        SimulationStruct** simulations, int ignoreAdetection);

extern int Write_Simulation_Results(SimState* HostMem,
        SimulationStruct* sim, clock_t simulation_time);

extern void FreeSimulationStruct(SimulationStruct* sim, int n_simulations);

#endif  // _CUDAMCML_H_
