#ifndef _CUDAMCML_H_
#define _CUDAMCML_H_

// Use the Mersenne Twister random number generator
// (as opposed to Tausworthe generator)
#define USE_MT_RNG

/////////////////////////////
//MCML constants
/////////////////////////////

#define WEIGHT 1E-4F        /* Critical weight for roulette. */

// scaling factor for photon weight, which is then converted to integer
#define WEIGHT_SCALE 12000000

#define PI_const 3.1415926F
#define RPI 0.318309886F

//NOTE: ********#define COSZERO (1.0-1.0E-12) infers double data type
#define COSNINETYDEG 1.0E-6F
#define COSZERO (1.0F - 1.0E-6F)    // NOTE: 1.0-1.0E-7 doesn't work in CUDA!
#define CHANCE 0.1F

#define STR_LEN 200

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

typedef struct
{
    float z_min;		// Layer z_min [cm]
    float z_max;		// Layer z_max [cm]
    float mutr;			// Reciprocal mu_total [cm]
    float mua;			// Absorption coefficient [1/cm]
    float g;			// Anisotropy factor [-]
    float n;			// Refractive index [-]
} LayerStruct;

typedef struct
{
    float dr;		// Detection grid resolution, r-direction [cm]
    float dz;		// Detection grid resolution, z-direction [cm]

    int na;			// Number of grid elements in angular-direction [-]
    int nr;			// Number of grid elements in r-direction
    int nz;			// Number of grid elements in z-direction
} DetStruct;

typedef struct 
{
    char outp_filename[STR_LEN];
    char inp_filename[STR_LEN];

    // the starting and ending offset (in the input file) for this simulation
    long begin, end;
    // ASCII or binary output
    char AorB;

    unsigned long number_of_photons;
    int ignoreAdetection;
    float start_weight;

    DetStruct det;

    unsigned int n_layers;
    LayerStruct* layers;
} SimulationStruct;

// Per-GPU simulation states
// One instance of this struct exists in the host memory, while the other
// in the global memory.
typedef struct
{
    // points to a scalar that stores the number of photons that are not
    // completed (i.e. either on the fly or not yet started)
    unsigned int *n_photons_left;

    // per-thread seeds for random number generation
    // arrays of length NUM_THREADS
    // We put these arrays here as opposed to in GPUThreadStates because
    // they live across different simulation runs and must be copied back
    // to the host.
#ifdef USE_MT_RNG
    unsigned long long *x;
    unsigned int *a;
#else
    unsigned int *s1;
    unsigned int *s2;
    unsigned int *s3;
#endif

    // output data
    unsigned long long* Rd_ra;
    unsigned long long* A_rz;			// Pointer to the 2D detection matrix!
    unsigned long long* Tt_ra;
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
    // GPU-specific constant parameters:
    // the limit that indicates overflow of an element of A_rz
    // in the shared memory
    unsigned int A_rz_overflow;

} HostThreadState;

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

extern void print_usage();

extern int interpret_arg(int argc, char* argv[], unsigned long long* seed,
        int* ignoreAdetection, unsigned int *num_GPUs);

extern int read_simulation_data(char* filename,
        SimulationStruct** simulations, int ignoreAdetection);

extern int Write_Simulation_Results(SimState* HostMem,
        SimulationStruct* sim, double simulation_time);

extern void FreeSimulationStruct(SimulationStruct* sim, int n_simulations);

#endif  // _CUDAMCML_H_
