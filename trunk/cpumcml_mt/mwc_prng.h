float rand_MWC_co(unsigned long long* x)
{
const unsigned int a = 4294967118;
		//Generate a random number [0,1)
		*x=(*x&0xffffffffull)*a+(*x>>32);
		return (float)((unsigned int)(*x)/(float)0x100000000);

// The typecast will truncate the x so that it is 0<=x<(2^32-1),__uint2float_rz ensures a round towards zero since 32-bit floating point cannot represent all integers that large. Dividing by 2^32 will hence yield [0,1)

}//end __device__ rand_MWC_co
