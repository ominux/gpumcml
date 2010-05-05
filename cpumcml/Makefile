CFLAGS =-O3
CC=cc
RM=/bin/rm -rf
LOCAL_LIBRARIES=-L/usr/lib -lm

PROFILE =
#PROFILE =
OBJS = mcmlmain.o mcmlgo.o mcmlio.o mcmlnr.o
.c.o:
	$(RM) $@
	$(CC) -c $(PROFILE) $(CFLAGS) $*.c
#####
all : mcml
mcml: $(OBJS)
	$(RM) $@
	$(CC) -o   $@ $(OBJS) $(PROFILE) $(LOCAL_LIBRARIES)
clean::
	$(RM) mcml
	$(RM) *.o

