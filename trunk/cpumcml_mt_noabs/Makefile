CFLAGS = -O3 -fno-strict-aliasing -DDSFMT_MEXP=19937 --param max-inline-insns-single=1800 --param inline-unit-growth=500 --param large-function-growth=900
CC=gcc
RM=/bin/rm -rf
LOCAL_LIBRARIES= -lm
OBJS = mcmlmain.o mcmlgo.o mcmlio.o mcmlnr.o dSFMT.o

.c.o:
	 $(RM) $@
	 $(CC) -c $(CFLAGS) $*.c
#####


all : mcml 


mcml: $(OBJS)
	 $(RM) $@
	 $(CC)  -o  $@ $(OBJS) $(LOCAL_LIBRARIES)


clean::
	 $(RM) mcml
	 $(RM) mcml*.o

