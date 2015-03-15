


## Introduction to MCML ##

---

The MCML algorithm models steady-state light transport in multi-layered turbid media
using the Monte Carlo method. A pencil beam perpendicular to the surface is modelled; more complex
sources can be modelled with modifications or with other tools. The implementation
assumes infinitely wide layers, each described by its thickness and its optical properties, comprising the absorption coefficient, scattering coefficient, anisotropy factor, and refractive index.

In the MCML code, three physical quantities – absorption, reflectance, and transmittance
– are calculated in a spatially-resolved manner. Absorption is recorded in a 2-D array, which stores the photon absorption probability density as a function of radius r and depth z for the pencil beam (or impulse response). Absorption probability density can be converted into more common quantities, such as photon fluence by dividing by the local absorption coefficient.

The simulation of each photon packet consists of a repetitive sequence of computational steps and can be made independent of other photon packets by creating separate absorption arrays and, importantly, decoupling random number generation using different seeds. Therefore, a conventional software-based acceleration approach involves processing photon packets simultaneously on multiple processors. The following figure shows a flow chart of the key steps in an MCML simulation, which includes photon initialization, position update, direction update, fluence update, and photon termination. Further details on each computational step may be found in the original papers by [Wang et al.](http://omlc.ogi.edu/software/mc/mcpubs/1995LWCMPBMcml.pdf)

![https://gpumcml.googlecode.com/svn/wiki/images/mcml.jpg](https://gpumcml.googlecode.com/svn/wiki/images/mcml.jpg)

## Parallelization of MCML ##

---

One important difference between writing CUDA code and writing a traditional C program (for
sequential execution on a CPU) is the need to devise an efficient parallelization scheme for the
case of CUDA programming. Although the syntax used by CUDA is, in theory, very similar
to C, the programming approach differs significantly. Compared to serial execution on a single
CPU where only one photon packet is simulated at a time, the GPU-accelerated version can
simulate many photon packets in parallel using multiple threads executed across many scalar
processors. The total number of photon packets to be simulated are split equally among the
threads.

## GPU program/kernel ##

---

The GPU program or kernel contains the computationally intensive loop in the MCML simulation. Other miscellaneous
tasks, such as reading the simulation input file, are performed on the host CPU. Each thread
executes the same loop, except using a unique random number sequence. Also, a single copy
of the absorption array is allocated in the global memory, and all the threads update this array
concurrently using atomic instructions. Although it is, in theory, possible to allocate a private
copy of the array for each thread, this approach greatly limits the number of threads that can be
launched when the absorption array is large, especially in 3-D cases. Therefore, a single copy
is allocated by default (and the program provides an option to specify the number of replicas).

## GPU kernel configuration ##

---

The kernel configuration, which significantly affects performance, is set based on the generation
of the GPU detected at runtime. The number of thread blocks is chosen based on the
number of SMs detected, while the number of threads within each block is chosen based on the
Compute Capability of the graphics card. For example, the kernel configuration is specified as
15 thread blocks (Q=15), each containing 896 threads (P=896), for GTX 480 with a Compute
Capability of 2.0. As shown below, each thread block is physically mapped onto one of the 15
multiprocessors and the 896 threads interleave its execution on the 32 scalar processors within
each multiprocessor on GTX 480.

![https://gpumcml.googlecode.com/svn/wiki/images/gpumcml.png](https://gpumcml.googlecode.com/svn/wiki/images/gpumcml.png)