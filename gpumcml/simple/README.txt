=======================================================
|| README 
=======================================================

|| A1) FEATURES
-------------------------------------------------------
- Supports Fermi GPU architecture 
- Backward compatible on pre-Fermi graphics cards
- Supports linux and Windows environment (Visual Studio)

|| A2) COMPARISON WITH OPTIMIZED GPUMCML
-------------------------------------------------------
- No optimization on reducing/hiding atomic access time
  to the global memory
  > caching of high fluence region in shared memory 
  > storing consecutive absorption at the same memory
    address in register
  > creating multiple copies of the absorption array
    in the global memory to reduce contention of
    atomic accesses
- No multi-GPU support 

|| A3) DOWNLOAD CUDA 3.0, Drivers and SDK
-------------------------------------------------------
1) Go to http://developer.nvidia.com/object/cuda_3_0_downloads.html
2) Download and install the following: 
   a) Developer Drivers
   b) CUDA Toolkit
   c) GPU Computing SDK

|| B) LINUX TEST RUN 
-------------------------------------------------------
1) Type make 

2) Execute the correct version based on the compute 
   capability (CC) of your graphics card.  Check 
   Appendix A of the CUDA programming guide 
   (http://developer.download.nvidia.com/compute/cuda/
    3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf) 

   GTX 480 (Fermi - CC 2.0) 
   Type ./gpumcml.sm_20 input/test.mci 

   GTX 280 (pre-Fermi - CC 1.3) 
   ./gpumcml.sm_13 input/test.mci 

3) Use the MATLAB scripts to plot the output 
   a) Go to the "viewoutput" folder
   b) Run plot_simulation.m 
   c) Select the output file when prompted

|| C) WINDOWS TEST RUN (Visual Studio or VS 2008)
-------------------------------------------------------
1) Open the project file with VS 2008: GPUMCML.vcproj 

2) Click Build Solution (or press F7) 

3) Run the program in VS (press F5)  OR 

   Run the program in command prompt 
   a) Start-->Run-->Type cmd
   b) Type cd "<DIR>/simple_gpumcml/executable" where 
      <DIR> is the absolute path where you put the files
   b) Launch the program: GPUMCML ../input/test.mci

* Two output files are generated in the "executable" folder: 
  - test_1M.mco
  - test_100K.mco

4) Use the MATLAB scripts to plot the output 
   a) Go to the "viewoutput" folder
   b) Run plot_simulation.m 
   c) Select 1 of the 2 output files stored in the "executable" folder when prompted

