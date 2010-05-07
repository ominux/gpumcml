CFLAGS = -O3
CC=cc
RM=/bin/rm -rf
LOCAL_LIBRARIES= -lm
OBJS = mcmlmain.o mcmlgo.o mcmlio.o mcmlnr.o 

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

