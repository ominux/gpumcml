/***********************************************************
 *  Copyright Univ. of Texas M.D. Anderson Cancer Center
 *  1992.
 *
 *	main program for Monte Carlo simulation of photon
 *	distribution in multi-layered turbid media.
 *
 ****/

/****
 *	THINKCPROFILER is defined to generate profiler calls in 
 *	Think C. If 1, remember to turn on "Generate profiler 
 *	calls" in the options menu. 
 ****/
#define THINKCPROFILER 0	

/* GNU cc does not support difftime() and CLOCKS_PER_SEC.*/
#define GNUCC 0

#if THINKCPROFILER
#include <profile.h>
#include <console.h>
#endif

#include <string.h>
//#include <stdio.h>
//#include <stdlib.h>
#include "mcml.h"
#include "hrtime.h" 
#include <sched.h>    //Used to set Number of processors useda

/*	Declare before they are used in main(). */
FILE *GetFile(char *);
short ReadNumRuns(FILE*);
void ReadParm(FILE*, InputStruct *);
void CheckParm(FILE*, InputStruct *);
void InitOutputData(InputStruct, OutStruct *);
void FreeData(InputStruct, OutStruct *);
double Rspecular(LayerStruct *);
void LaunchPhoton(double, LayerStruct *, PhotonStruct *);
void HopDropSpin(InputStruct *, PhotonStruct *, OutStruct *, int);
void SumScaleResult(InputStruct, OutStruct *);
void WriteResult(InputStruct, OutStruct, char *);

//>>>>>>>>>>>>>>>>Multi-Threading>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
void * DoOneThread(void *);
void SumOutPtr(OutStruct *sum_out_parm, OutStruct Nout_parm, short nz,
    short nr, short na);

/***********************************************************
 *	If F = 0, reset the clock and return 0.
 *
 *	If F = 1, pass the user time to Msg and print Msg on 
 *	screen, return the real time since F=0. 
 *
 *	If F = 2, same as F=1 except no printing.  
 *
 *	Note that clock() and time() return user time and real 
 *	time respectively.
 *	User time is whatever the system allocates to the 
 *	running of the program; 
 *	real time is wall-clock time.  In a time-shared system,
 *	they need not be the same.
 *	
 *	clock() only hold 16 bit integer, which is about 32768 
 *	clock ticks.
 ****/
time_t PunchTime(char F, char *Msg) {
#if GNUCC
  printf ("Inside GNUCC statement\n");
  return(0);
#else
  static clock_t ut0; /* user time reference. */
  static time_t rt0; /* real time reference. */
  long double secs;
  char s[STRLEN];

  if (F==0) {
    ut0 = clock();
    rt0 = time(NULL);

    return (0);
  } else if (F==1) {
    secs = (clock() - ut0)/(double)CLOCKS_PER_SEC;

    if (secs<0)
      secs=0; /* clock() can overflow. */
    sprintf(s, "User time: %8.0lf sec = %8.2lf hr.  %s\n", secs, secs /3600.0,
        Msg);
    puts(s);
    strcpy(Msg, s);
    return (difftime(time(NULL), rt0));
  } else if (F==2) {
    return (difftime(time(NULL), rt0));
  } else
    return (0);
#endif
}

/***********************************************************
 *	Print the current time and the estimated finishing time.
 *
 *	P1 is the number of computed photon packets.
 *	Pt is the total number of photon packets.
 ****/
void PredictDoneTime(long P1, long Pt) {
  time_t now, done_time;
  struct tm *date;
  char s[80];

  now = time(NULL);
  date = localtime(&now);
  strftime(s, 80, "%H:%M %x", date);
  printf("Now %s, ", s);

  done_time = now + (time_t) (PunchTime(2, "")/(double)P1*(Pt-P1));
  date = localtime(&done_time);
  strftime(s, 80, "%H:%M %x", date);
  printf("End %s\n", s);
}

/***********************************************************
 *	Report time and write results. 
 ****/
void ReportResult(InputStruct In_Parm, OutStruct Out_Parm) {
  char time_report[STRLEN];

  strcpy(time_report, " Simulation time of this run.");
  //PunchTime(1, time_report);

  SumScaleResult(In_Parm, &Out_Parm);
  WriteResult(In_Parm, Out_Parm, time_report);
}

/***********************************************************
 *	Get the file name of the input data file from the 
 *	argument to the command line.
 ****/
void GetFnameFromArgv(int argc, char * argv[], char * input_filename) {
  if (argc>=2) { /* filename in command line */
    strcpy(input_filename, argv[1]);
  } else
    input_filename[0] = '\0';
}

/***********************************************************
 *	Execute Monte Carlo simulation for one independent run.
 ****/
void DoOneRun(short NumRuns, InputStruct *In_Ptr) {
  int i;
  OutStruct sum_out_parm;
  long num_photons = In_Ptr->num_photons;

#if THINKCPROFILER
  InitProfile(200,200); cecho2file("prof.rpt",0, stdout);
#endif

  //>>>>>>>>>>>>>>Performance Measurement 
  start_time = getElapsedTime();
  start_cycle = get_hrcycles();

  //>>>>>>>>>>>>>>>>Multi-Threading>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
  GlobalIn_Ptr=In_Ptr; //All threads share this input file pointer struct
  initRandom();

  pthread_t thread[NTHREAD];
  printf("Number of threads=%lu \n", NTHREAD);

  for (i=0; i<NTHREAD; i++)
    pthread_create(&thread[i], NULL, (void *) DoOneThread, (void *) i);
  //>>>>>>>>>>>>>>>>Multi-Threading>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#if THINKCPROFILER
  exit(0);
#endif

  for (i=0; i<NTHREAD; i++)
    pthread_join(thread[i], NULL);

  printf("After pthread_join\n");

  InitOutputData(*In_Ptr, &sum_out_parm);
  sum_out_parm.Rsp = Rspecular(In_Ptr->layerspecs);

  for (i=0; i<NTHREAD; i++)
    SumOutPtr(&sum_out_parm, out_parm[i], In_Ptr->nz, In_Ptr->nr, In_Ptr->na);

  end_cycle = get_hrcycles();
  end_time = getElapsedTime();

  printf("DoOneRun took %lf seconds \n", end_time - start_time);
  printf("DoOneRun took %lld cycles \n", end_cycle - start_cycle);

  ReportResult(*In_Ptr, sum_out_parm);
  FreeData(*In_Ptr, &sum_out_parm);
}

void SumOutPtr(OutStruct *sum_out_parm, OutStruct Nout_parm, short nz,
    short nr, short na) {
  //sum every items. 
  short iz, ir, ia; /* index to z & r. */

  for (ir=0; ir<nr; ir++)
    for (ia=0; ia<na; ia++) {
      sum_out_parm->Rd_ra[ir][ia] +=Nout_parm.Rd_ra[ir][ia];
      sum_out_parm->Tt_ra[ir][ia] +=Nout_parm.Tt_ra[ir][ia];
    }

  for (ir=0; ir<nr; ir++)
    for (iz=0; iz<nz; iz++)
      sum_out_parm->A_rz[ir][iz]+=Nout_parm.A_rz[ir][iz];
}

void * DoOneThread(void *i) {
  int pid = (int)i;
  register long i_photon=(long)GlobalIn_Ptr->num_photons/NTHREAD/NUM_NODE; //Nphotons;
  /* index to photon. register for speed.*/
  //TODO: Error-check to make sure all num_photons are used !!!

  //printf("Number of photons=%lu \n", i_photon);

  long photon_rep=10;
  PhotonStruct photon;
  //printf("pthread id %d \n", pid);

  InitOutputData(*GlobalIn_Ptr, &out_parm[pid]);
  out_parm[pid].Rsp = Rspecular(GlobalIn_Ptr->layerspecs);

  do {
    LaunchPhoton(out_parm[pid].Rsp, GlobalIn_Ptr->layerspecs, &photon);
    do
      HopDropSpin(GlobalIn_Ptr, &photon, &out_parm[pid], pid);
    while (!photon.dead);
  } while (--i_photon);
}

//>>>>>>>>>>>>>Distributed Computing Implementation >>>>>>>>>>>>>>>>>>>>>
void getClusterParam(int argc, char *argv[]) {
  //TODO: Make this dynamic 
  //  if (argc<3) { 
  //	perror ("Missing Parameter: ./mcml <inputFile> <NTHREAD> <CURRENT_NODE>(optional) <NUM_NODE> (optional)\n"); 
  //  	exit (-1); 
  //  }
  //  NTHREAD=(int)argv[2]; 

  if (argc>2) { //Distributed Computing 
    NUM_NODE=atoi(argv[3]);
    CURRENT_NODE=atoi(argv[2]);
    printf("NUM_NODE=%d\n", NUM_NODE);
    printf("CURRENT_NODE=%d\n", CURRENT_NODE);

    //TODO: Error check for range
  } else { //Default - 1 Node
    printf("Defaulting to 1 Node\n");
    NUM_NODE=1;
    CURRENT_NODE=0;
  }
}
//>>>>>>>>>>>>>Distributed Computing Implementation >>>>>>>>>>>>>>>>>>>>>

int setMask(int Nproc) {
  switch (Nproc) {
  case 1:
    printf("Using 1 proc \n");
    return 1;
    break;
  case 2:
    printf("Using 2 proc \n");
    return 3;
    break;
  case 3:
    printf("Using 3 proc \n");
    return 7;
    break;
  case 4:
    printf("Using 4 proc \n");
    return 15;
    break;
  default:
    printf("Default: Using all 4 proc \n");
    return 15;
    break;
  }
}
//>>>>>>>>>>>>>Setting Number of processors Used>>>>>>>>>>>>>>>>>>>>>>>>
int setNproc(int argc, char *argv[]) {
  int Nproc=1;

  if (argc>4) {
    Nproc=atoi(argv[4]);
    unsigned long mask;
    unsigned int len=sizeof(mask);
    if (sched_getaffinity(0, len, &mask) <0) {
      perror("sched_getaffinity");
      return -1;
    }

    printf("My affinity mask is : %081x\n", mask);

    mask=setMask(Nproc); 

    len=sizeof(mask);
    if (sched_setaffinity(0, len, &mask)<0) {
      perror("sched_setaffinity");
      return -1;
    }

    printf("My new affinity mask is: %081x\n", mask);
  } else { //Deafult- No change in mask
    printf("Defaulting to using ALL available processors (cores)\n");
  }
}

/***********************************************************
 *	The argument to the command line is filename, if any.
 *	Macintosh does not support command line.
 ****/
int main(int argc, char *argv[]) {
  int i;
  char input_filename[STRLEN];
  FILE *input_file_ptr;

  short num_runs; /* number of independent runs. */
  InputStruct in_parm;

  ShowVersion("Version 1.2, 1993");

  //>>>>>>>>>>>>>Setting Number of processors Used>>>>>>>>>>>>>>>>>>>>>>>>
  //Does not work on UHN cluster: Linux version 2.4.22-openmosix-3 
  //(gcc version 3.2.2 20030222 (Red Hat Linux 3.2.2-5)) #1 SMP
  //Works on Linux version 2.6.17-2-amd64 (Debian 2.6.17-9) 
  //(gcc version 4.1.2 20060901 (prerelease) (Debian 4.1.1-13)) #1 SMP 

//  if(setNproc(argc, argv)==-1) {
//    perror("Setting Number of Proc\n");
//    return -1; 
//  }
  //>>>>>>>>>>>>>

  //>>>>>>>>>>>>>Distributed Computing Implementation >>>>>>>>>>>>>>>>>>>>>
  getClusterParam(argc, argv);
  //>>>>>>>>>>>>>Distributed Computing Implementation >>>>>>>>>>>>>>>>>>>>>

  GetFnameFromArgv(argc, argv, input_filename);
  input_file_ptr = GetFile(input_filename);

  CheckParm(input_file_ptr, &in_parm);
  num_runs = ReadNumRuns(input_file_ptr);

  while (num_runs--) {
    ReadParm(input_file_ptr, &in_parm);
    DoOneRun(num_runs, &in_parm);
  }

  fclose(input_file_ptr);

  return (0);
}
