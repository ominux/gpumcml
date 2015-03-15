

## Overview of Directory ##

---

The following chart shows how the folders are organized and the purpose of each folder.  The same structure is used for both the fast version and the simplified version.  The source code files are stored in the top-level directory.

![https://gpumcml.googlecode.com/svn/wiki/images/directory.jpg](https://gpumcml.googlecode.com/svn/wiki/images/directory.jpg)

## Source File Description ##

---

### Common to all platforms ###
  * **gpumcml.h**: Header file for common data structures and constants (CPU and GPU)
  * **gpumcml\_kernel.h**: Header file for GPU-related data structures and kernel configurations
  * **gpumcml\_main.cu**: Source file containing the main function and host CPU code
  * **gpumcml\_kernel.cu**: Source file for GPU-MCML kernel functions (simulation functions)
  * **gpumcml\_mem.cu**: Source file for GPU memory allocation, initialization, and transfer between the host and GPU
  * **gpumcml\_rng.cu**: Source file for the Random Number Generator algorithm and initialization
  * **gpumcml\_io.c**: Source file for command line and input parameter parsing + simulation file output


### For Linux Only ###
  * **makefile**: code compilation

### For Windows Only ###
  * **GPUMCML.vcproj**: Visual Studio 2008 project file
  * **GPUMCML.sln**: Visual Studio 2008 solution file
  * **GPUMCML.vcproj.user**: Visual Studio 2008 user file (stores the execution/debugging user options)