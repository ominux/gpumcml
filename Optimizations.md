

## 1. Programming Graphics Processing Units ##

---

This section introduces the key terminology for understanding the NVIDIA GPU hardware and
its programming model. This learning curve is required to fully utilize this emerging scientific
computing platform for this and other related applications.

### 1.1 CUDA ###
GPU-accelerated scientific computing is becoming increasingly popular with the release of an
easier-to-use programming model and environment from NVIDIA (Santa Clara, CA), called
[CUDA](http://www.nvidia.com/object/cuda_home_new.html), short for Compute Unified Device Architecture. CUDA provides a C-like programming
interface for NVIDIA GPUs and it suits general-purpose applications much better
than traditional GPU programming languages. However, performance optimization of a CUDA
program requires careful consideration of the GPU architecture.

In CUDA, the host code and the device code are written in a single program. The former
is executed sequentially by a single thread on the CPU, while the latter is executed in parallel
by many threads on the GPU. Here, a thread is an independent execution context that can, for
example, simulate an individual photon packet in the MCML algorithm. The device code is
expressed in the form of a kernel function. It is similar to a regular C function, except that it
specifies the work of each GPU thread, parameterized by a thread index variable. Correspondingly,
a kernel invocation is similar to a regular function call except that it must specify the
geometry of a grid of threads that executes the kernel, also referred to as the kernel configuration.
A grid is first composed of a number of thread blocks, each of which then has a number
of threads.

### 1.2 NVIDIA GPU Architecture ###
CUDA-enabled NVIDIA GPUs have gone through two generations; an example for each generation
includes Geforce 8800 GTX and Geforce GTX 280, respectively. The third generation,
named Fermi and represented by Geforce GTX 480, was released recently (in the second quarter
of 2010). The following figure compares the underlying hardware architecture of GTX 280 and GTX
480, showing both a unique processor layout and memory hierarchy.

![https://gpumcml.googlecode.com/svn/wiki/images/gpu.jpg](https://gpumcml.googlecode.com/svn/wiki/images/gpu.jpg)

GTX 280 has 30 streaming multiprocessors (SMs), each with 8 scalar processors (SPs). Note
that the 240 SPs (total) are not 240 independent processors; instead, they are 30 independent
processors that can perform 8 similar computations at a time. From the programmer’s perspective,
each thread block is assigned to an SM, and each thread within it is further scheduled
to execute on one of the SPs. Compared to GTX 280, the Fermi-based GTX 480 has half the
number of SMs, but each SM contains four times the number of SPs, resulting in a total of 480
SPs.
Apart from the processor layout, the programmer must understand the different layers and
types of memory on the graphics card, due to the significant difference in memory access
time. At the bottom layer resides the off-chip device memory (also known as global memory),
which is the largest yet slowest type of GPU memory. Closer to the GPU are various
kinds of fast on-chip memories. Common to both GTX 280 and GTX 480 are registers at the
fastest speed, shared memory at close to register speed, and a similarly fast cache for constant
memory (for read-only data). On-chip memories are roughly a hundred times faster than the
off-chip global memory, but they are very limited in storage capacity. Finally, there is a region
in device memory called local memory for storing large data structures, such as arrays, which
cannot be mapped into registers by the compiler. Compared to GTX 280, GTX 480 has up to
triple the amount of shared memory for each SM. Most importantly, it includes two levels of
hardware-managed caches to ameliorate the large difference in access time of on-chip and offchip
memories. At the bottom, there is an L2 cache for the global memory with roughly half
the access time. At the top, there is an L1 cache within each SM, at the speed of the shared
memory. In fact, the L1 cache and the shared memory are partitions of the same hardware. The
programmer can choose one of two partitioning schemes for each kernel invocation. Even with
these improvements in GTX 480, it is still important to map the computation efficiently to the
different types of memories for high performance.

### 1.3 Atomic Instructions ###
CUDA also provides a mechanism to synchronize the execution of threads using atomic instructions,
which coordinate sequential access to a shared variable (such as the absorption array in
the MCML code). Atomic instructions guarantee data consistency by allowing only one thread
to update the shared variable at any time; however, in doing so, it stalls other threads that require
access to the same variable. As a result, atomic instructions are much more expensive than regular
memory operations. Although their speed has been improved on Fermi-based GPUs,
it is still very important to optimize atomic accesses to global memory, as explained in the
following section.


## 2. GPU-accelerated MCML Code (GPU-MCML) ##

---

This section presents the implementation details of the GPU-accelerated MCML program
(named GPU-MCML), highlighting how a high level of parallelism is achieved, while avoiding
memory bottlenecks caused by atomic instructions and global memory accesses. The need to
carefully consider the underlying GPU architecture, particularly the differences between the pre-Fermi and Fermi GPUs, is discussed.

### 2.1 Key Performance Bottleneck ###
As all threads need to atomically access the same absorption array in the global memory during
every fluence update step, this step becomes a major performance bottleneck when thousands
of threads are present. In CUDA, atomic addition
is performed using the atomicAdd() instruction. However, using atomicAdd() instructions
to access the global memory is particularly slow, both because global memory access is
a few orders of magnitude slower than that of on-chip memories and because atomicity prevents
parallel execution of the code (by stalling other threads in the code segment where atomic
instructions are located). This worsens with increasing number of threads due to the higher
probability for simultaneous access to an element, also known as contention.

### 2.2 Solution to Performance Bottleneck ###
To reduce contention and access time to the Arz array, two memory optimizations were
applied:

  1. **Storing the high-fluence region in shared memory**: The first optimization is based on the high access rate of the Arz elements near the photon source (or at the origin in the MCML model), causing significant contention when atomic instructions are used. Therefore, this region of the Arz array is cached in the shared memory. Namely, if a photon packet is absorbed inside the high-fluence region, its weight is accumulated atomically into the shared memory copy. Otherwise, its weight is added directly into the global memory atomically. At the end of the simulation, the values temporarily stored in the shared memory copy are written (flushed) to the master copy in the global memory.
  1. **Caching photon absorption history in registers**: To further reduce the number of atomic accesses (even to the shared memory), the recent write history, representing previous absorption events, can be stored privately in registers by each thread.

As an additional optimization to avoid atomic accesses, in the GPU version, photon packets at
locations beyond the coverage of the absorption grid (as specified through the input parameters
dr, dz, nr, and nz) no longer accumulate their weights at the perimeter of the grid, unlike
in the original MCML code. Note that these boundary elements were known to give invalid
values in the original MCML code.

### 2.3 Alternative Solution: Reflectance/transmittance-only Mode ###
GPU-MCML offers an alternative way to overcome the performance bottleneck. In some cases,
the user is not interested in the internal absorption probability collected in the Arz array. As
the major performance bottleneck is caused by the atomic update of the Arz array for almost
every step for each photon, ignoring the array update (while still reducing the photon packet
weight after each scattering event) causes a major performance boost while still retaining valid
reflectance and transmittance outputs. The maximum number of atomic memory operations
is reduced to the total number of photon packets launched (since reflectance/transmittance is
recorded atomically only upon exiting the tissue, which happens once per photon packet) and
these operations are spread out over the entire simulation. Also, the small number of memory
operations compared to arithmetic operations allows the GPU to “hide” the cost of memory
operations.