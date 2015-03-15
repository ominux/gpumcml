

This tutorial aims to guide new users through the initial pain of setting up, compiling and running the source code.

## Step 0: Check your graphics card ##

---

Before you start, make sure that your graphics card is CUDA compatible (check Appendix A of the [CUDA programming guide](http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf)) and has a Compute Capability of between 1.1 and 2.0 (Fermi).  Note that atomic instructions are only supported on devices with a Compute Capability of at least 1.1.  Most graphics cards today should be in this category.

If you don't have a proper graphics card or if you want the best performance from our code, we recommend purchasing the latest Fermi-generation GPU, such as [GeForce GTX 480](http://www.nvidia.com/object/product_geforce_gtx_480_us.html).  However, ensure that your power supply is at least 600 W (with 6-pin & 8-pin power connectors) and the motherboard supports PCI-E 2.0 x 16.

## Step 1a: Install Visual Studio (Windows Users Only) ##

---

Install Microsoft Visual Studio 2008 (Visual C++ 2008). For a free version, download [Visual C++ 2008 Express Edition](http://www.microsoft.com/express/downloads/).

## Step 1b: Install GNU Compiler and Utilities (Linux Users Only) ##

---

Install the GCC compiler and the make utility, from the package manager of your choice. On a Debian-derived OS (e.g., Ubuntu), simply type the following command:

`> sudo aptitude install build-essential`

## Step 2: Install CUDA, Developer Drivers, and SDK Samples ##

---

Then, just follow all the steps in the [NVIDIA Getting Started Guide](http://developer.download.nvidia.com/compute/cuda/3_0/docs/GettingStartedWindows.pdf).

If you have a Fermi GPU, install [CUDA ToolKit 3.0 and GPU Computing SDK code samples](http://developer.nvidia.com/object/cuda_3_0_downloads.html) along with the proper Developer Drivers.  **Make sure you upgrade both the driver and the CUDA Toolkit, and not one of the two!  Otherwise, your code will not run !**

## Step 3: Verify the steps above ##

---

Make sure you have everything set up properly, by running a simple SDK code sample, as suggested in the [NVIDIA Getting Started Guide](http://developer.download.nvidia.com/compute/cuda/3_0/docs/GettingStartedWindows.pdf)

## Step 4: Download our source code package ##

---

Download and unarchive: [Fast GPUMCML](http://gpumcml.googlecode.com/files/fast-gpumcml.zip) OR [Simple GPUMCML](http://gpumcml.googlecode.com/files/simple_gpumcml.zip).

In Linux, type **unzip fast-gpumcml.zip** or **unzip simple-gpumcml.zip**.   In Windows, use any one of the common compression software such as WinRAR or [7-Zip (free)](http://www.7-zip.org/).

## Step 5: Compile and run GPUMCML ##

---

#### _For Windows User (Visual Studio or VS 2008)_ ####
  1. Open the project file with VS 2008: **GPUMCML.vcproj**
  1. Click Build Solution (or press F7)
  1. Run the program in VS (press F5) _OR_ run the program using the command prompt:
    * Start-->Run-->Type **cmd**
    * Type **cd DIR/fast\_gpumcml/executable** where **DIR** is the absolute path where you put the files
    * Launch the program: Type **GPUMCML ../input/test.mci**

#### _For Linux User (Tested on Ubuntu)_ ####
1) Type **make**

2) Execute the correct version based on the compute capability (CC) of your graphics card (check Appendix A of the [CUDA programming guide](http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf))

> GTX 480 (Fermi - CC 2.0): Type **./gpumcml.sm\_20 input/test.mci**

> GTX 280 (CC 1.3): Type **./gpumcml.sm\_13 input/test.mci**


## Step 6: Plot the simulation output with MATLAB ##

---

Use the MATLAB scripts to plot the output
  * Go to the **viewoutput** folder
  * Run **plot\_simulation.m**
  * Select the output file when prompted.  Two files are generated in Step 5: **test\_1M.mco** and **test\_100K.mco** executed with 1 million and 100 thousand photon packets, respectively (the files are in the **executable** folder if VS 2008 is used or in the top-level directory if Linux is used).