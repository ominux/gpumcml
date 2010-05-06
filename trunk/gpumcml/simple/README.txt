=======================================================
|| README - Simple GPUMCML
=======================================================

Features: 
- No atomic access optimization 
  > caching of high fluence region in shared memory 
  > storing consecutive absorption at the same mem address
    in register  
- No multi-GPU support 
- Simplified function parameter list using PhotonStruct
- Supports Fermi architecture 
- Now backwards compatible (tested on GTX 280)
- Supports linux and Windows environment (Visual Studio 2008)


