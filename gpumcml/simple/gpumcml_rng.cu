/*****************************************************************************
*
* Random Number Generator Algorithm and Initialization
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

#include "gpumcml_kernel.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
//   Generates a random number between 0 and 1 [0,1) 
//////////////////////////////////////////////////////////////////////////////
__device__ FLOAT rand_MWC_co(UINT64* x,UINT32* a)
{
  *x=(*x&0xffffffffull)*(*a)+(*x>>32);
  return __fdividef(__uint2float_rz((UINT32)(*x)),(FLOAT)0x100000000);
  // The typecast will truncate the x so that it is 0<=x<(2^32-1),
  // __uint2FLOAT_rz ensures a round towards zero since 32-bit FLOATing point 
  // cannot represent all integers that large. 
  // Dividing by 2^32 will hence yield [0,1)
} 

//////////////////////////////////////////////////////////////////////////////
//   Generates a random number between 0 and 1 (0,1]
//////////////////////////////////////////////////////////////////////////////
__device__ FLOAT rand_MWC_oc(UINT64* x,UINT32* a)
{
  return 1.0f-rand_MWC_co(x,a);
} 

//////////////////////////////////////////////////////////////////////////////
//   Initialize random number generator 
//////////////////////////////////////////////////////////////////////////////
int init_RNG(UINT64 *x, UINT32 *a, 
             const UINT32 n_rng, const char *safeprimes_file, UINT64 xinit)
{
  FILE *fp;
  UINT32 begin=0u;
  UINT32 fora,tmp1,tmp2;

  if (strlen(safeprimes_file) == 0)
  {
    // Try to find it in the local directory
    safeprimes_file = "safeprimes_base32.txt";
  }

  fp = fopen(safeprimes_file, "r");

  if(fp == NULL)
  {
    printf("Could not find the file of safeprimes (%s)! Terminating!\n", safeprimes_file);
    return 1;
  }

  fscanf(fp,"%u %u %u",&begin,&tmp1,&tmp2);

  // Here we set up a loop, using the first multiplier in the file to generate x's and c's
  // There are some restictions to these two numbers:
  // 0<=c<a and 0<=x<b, where a is the multiplier and b is the base (2^32)
  // also [x,c]=[0,0] and [b-1,a-1] are not allowed.

  //Make sure xinit is a valid seed (using the above mentioned restrictions)
  if((xinit == 0ull) | (((UINT32)(xinit>>32))>=(begin-1)) | (((UINT32)xinit)>=0xfffffffful))
  {
    //xinit (probably) not a valid seed! (we have excluded a few unlikely exceptions)
    printf("%llu not a valid seed! Terminating!\n",xinit);
    return 1;
  }

  for (UINT32 i=0;i < n_rng;i++)
  {
    fscanf(fp,"%u %u %u",&fora,&tmp1,&tmp2);
    a[i]=fora;
    x[i]=0;
    while( (x[i]==0) | (((UINT32)(x[i]>>32))>=(fora-1)) | (((UINT32)x[i])>=0xfffffffful))
    {
      //generate a random number
      xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);

      //calculate c and store in the upper 32 bits of x[i]
      x[i]=(UINT32) floor((((double)((UINT32)xinit))/(double)0x100000000)*fora);//Make sure 0<=c<a
      x[i]=x[i]<<32;

      //generate a random number and store in the lower 32 bits of x[i] (as the initial x of the generator)
      xinit=(xinit&0xffffffffull)*(begin)+(xinit>>32);//x will be 0<=x<b, where b is the base 2^32
      x[i]+=(UINT32) xinit;
    }
    //if(i<10)printf("%llu\n",x[i]);
  }
  fclose(fp);

  return 0;
}

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

