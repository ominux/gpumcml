/*	This file is part of CUDAMCML.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/

#define NFLOATS 5
#define NINTS 5

#include <time.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>

#include "cudamcml.h"

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

void print_usage()
{
    printf("\n");
    printf("cudamcml <simulation input file> [-A] [-G<num_GPUs>]\n\n");
    printf("    -A: ignore absorption detection (no output of A_rz)\n"); 
    printf("    -G: specify the number of GPUs to be used\n");
    printf("\n");
}

int interpret_arg(int argc, char* argv[], unsigned long long* seed,
        int* ignoreAdetection, unsigned int *num_GPUs)
{
    int i;

    // Start from the 2nd argument (after the file name).
    for(i=2;i<argc;i++)
    {
        if(!strcmp(argv[i], "-A"))
        {
            *ignoreAdetection=1;
            printf("Ignoring A-detection!\n");
        }
        else if(!strncmp(argv[i],"-S",2) && sscanf(argv[i],"%*2c %llu",seed))
        {
            printf("Seed=%llu\n",*seed);
        }
        else if (strncmp(argv[i], "-G", 2) == 0)
        {
            sscanf(&argv[i][2], "%u", num_GPUs);
        }
        else
        {
            fprintf(stderr, "Unknown argument %s!\n", argv[i]);
            return 1;
        }
    }

    return 0;
}

int Write_Simulation_Results(SimState* HostMem, SimulationStruct* sim, double simulation_time)
{
	FILE* pFile_inp;
	FILE* pFile_outp;
	char mystring[STR_LEN];

	// Copy stuff from sim->det to make things more readable:
	double dr=(double)sim->det.dr;		// Detection grid resolution, r-direction [cm]
	double dz=(double)sim->det.dz;		// Detection grid resolution, z-direction [cm]
	double da=PI_const/(2*sim->det.na);		// Angular resolution [rad]?
	
	int na=sim->det.na;			// Number of grid elements in angular-direction [-]
	int nr=sim->det.nr;			// Number of grid elements in r-direction
	int nz=sim->det.nz;			// Number of grid elements in z-direction


	int rz_size = nr*nz;
	int ra_size = nr*na;
	int r,a,z;
	unsigned int l;
	int i;

	unsigned long long temp=0;
    double scale1 = (double)(WEIGHT_SCALE)*(double)sim->number_of_photons;
	double scale2;
	
	// Calculate and write RAT
	unsigned long long Rd=0;	// Diffuse reflectance [-]
	unsigned long long A=0;		// Absorbed fraction [-]
	unsigned long long T=0;		// Transmittance [-]
		
	// Open the input and output files
	pFile_inp = fopen (sim->inp_filename , "r");
	if (pFile_inp == NULL){perror ("Error opening input file");return 0;}

	pFile_outp = fopen (sim->outp_filename , "w");
	if (pFile_outp == NULL){perror ("Error opening output file");return 0;}

	// Write other stuff here first!

	fprintf(pFile_outp,"A1 	# Version number of the file format.\n\n");
	fprintf(pFile_outp,"####\n");
	fprintf(pFile_outp,"# Data categories include: \n");
	fprintf(pFile_outp,"# InParm, RAT, \n");
	fprintf(pFile_outp,"# A_l, A_z, Rd_r, Rd_a, Tt_r, Tt_a, \n");
	fprintf(pFile_outp,"# A_rz, Rd_ra, Tt_ra \n");
	fprintf(pFile_outp,"####\n\n");

	// Write simulation time
	fprintf(pFile_outp,"# User time: %.3f sec\n\n",simulation_time);


	fprintf(pFile_outp,"InParam\t\t# Input parameters:\n");
	// Copy the input data from inp_filename
	//printf("pos=%d\n",ftell(pFile_inp));
	fseek(pFile_inp, sim->begin, SEEK_SET);
	while(sim->end>ftell(pFile_inp))
	{
		//printf("pos=%d\n",ftell(pFile_inp));
		fgets(mystring , STR_LEN , pFile_inp);
		fputs(mystring , pFile_outp );
	}

	//printf("pos=%d\n",ftell(pFile_inp));
	fclose(pFile_inp);

	for(i=0;i<rz_size;i++)A+= HostMem->A_rz[i];
	for(i=0;i<ra_size;i++){T += HostMem->Tt_ra[i];Rd += HostMem->Rd_ra[i];}

	fprintf(pFile_outp,"\nRAT #Reflectance, absorption transmission\n");
	fprintf(pFile_outp,"%G \t\t #Specular reflectance [-]\n", 1.0F - sim->start_weight);
	fprintf(pFile_outp,"%G \t\t #Diffuse reflectance [-]\n",(double)Rd/scale1);
	fprintf(pFile_outp,"%G \t\t #Absorbed fraction [-]\n",(double)A/scale1);
	fprintf(pFile_outp,"%G \t\t #Transmittance [-]\n",(double)T/scale1);


	// Calculate and write A_l
	fprintf(pFile_outp,"\nA_l #Absorption as a function of layer. [-]\n");
	z=0;
	for(l=1;l<=sim->n_layers;l++)
	{
		temp=0;
		while(((double)z+0.5)*dz<=sim->layers[l].z_max)
		{
#if 0
			for(r=0;r<nr;r++) temp+=HostMem->A_rz[z*nr+r];
#else
            for (r = 0; r < nr; ++r) temp += HostMem->A_rz[r*nz + z];
#endif
			z++;
			if(z==nz)break;
		}
		fprintf(pFile_outp,"%G\n",(double)temp/scale1);
	}

	// Calculate and write A_z
	scale2=scale1*dz;
	fprintf(pFile_outp,"\nA_z #A[0], [1],..A[nz-1]. [1/cm]\n");
	for(z=0;z<nz;z++)
	{
		temp=0;
#if 0
		for(r=0;r<nr;r++) temp+=HostMem->A_rz[z*nr+r]; 
#else
        for (r = 0; r < nr; ++r) temp += HostMem->A_rz[r*nz + z];
#endif
		fprintf(pFile_outp,"%E\n",(double)temp/scale2);
	}

	// Calculate and write Rd_r
	fprintf(pFile_outp,"\nRd_r #Rd[0], [1],..Rd[nr-1]. [1/cm2]\n");
	for(r=0;r<nr;r++)
	{
		temp=0;
		for(a=0;a<na;a++) temp+=HostMem->Rd_ra[a*nr+r]; 
		scale2=scale1*2*PI_const*(r+0.5)*dr*dr;
		fprintf(pFile_outp,"%E\n",(double)temp/scale2);
	}

	// Calculate and write Rd_a 
	fprintf(pFile_outp,"\nRd_a #Rd[0], [1],..Rd[na-1]. [sr-1]\n");
	for(a=0;a<na;a++)
	{
		temp=0;
		for(r=0;r<nr;r++) temp+=HostMem->Rd_ra[a*nr+r]; 
		scale2=scale1*4*PI_const*sin((a+0.5)*da)*sin(da/2);
		fprintf(pFile_outp,"%E\n",(double)temp/scale2);
	}

	// Calculate and write Tt_r
	fprintf(pFile_outp,"\nTt_r #Tt[0], [1],..Tt[nr-1]. [1/cm2]\n");
	for(r=0;r<nr;r++)
	{
		temp=0;
		for(a=0;a<na;a++) temp+=HostMem->Tt_ra[a*nr+r];
		scale2=scale1*2*PI_const*(r+0.5)*dr*dr;
		fprintf(pFile_outp,"%E\n",(double)temp/scale2);
	}

	// Calculate and write Tt_a
	fprintf(pFile_outp,"\nTt_a #Tt[0], [1],..Tt[na-1]. [sr-1]\n");
	for(a=0;a<na;a++)
	{
		temp=0;
		for(r=0;r<nr;r++) temp+=HostMem->Tt_ra[a*nr+r]; 
		scale2=scale1*4*PI_const*sin((a+0.5)*da)*sin(da/2);
		fprintf(pFile_outp,"%E\n",(double)temp/scale2);
	}
	
	
	// Scale and write A_rz
	i=0;
	fprintf(pFile_outp,"\n# A[r][z]. [1/cm3]\n# A[0][0], [0][1],..[0][nz-1]\n# A[1][0], [1][1],..[1][nz-1]\n# ...\n# A[nr-1][0], [nr-1][1],..[nr-1][nz-1]\nA_rz\n");
	for(r=0;r<nr;r++)
	{
		for(z=0;z<nz;z++)
		{
			scale2=scale1*2*PI_const*(r+0.5)*dr*dr*dz;
#if 0
			fprintf(pFile_outp," %E ",(double)HostMem->A_rz[z*nr+r]/scale2);
#else
            fprintf(pFile_outp, " %E ", (double)HostMem->A_rz[r*nz + z] / scale2);
#endif
			if((i++)==4){i=0;fprintf(pFile_outp,"\n");}
		}
	}

	// Scale and write Rd_ra 
	i=0;
	fprintf(pFile_outp,"\n\n# Rd[r][angle]. [1/(cm2sr)].\n# Rd[0][0], [0][1],..[0][na-1]\n# Rd[1][0], [1][1],..[1][na-1]\n# ...\n# Rd[nr-1][0], [nr-1][1],..[nr-1][na-1]\nRd_ra\n");
	for(r=0;r<nr;r++)
	{
		for(a=0;a<na;a++)
		{
			scale2=scale1*2*PI_const*(r+0.5)*dr*dr*cos((a+0.5)*da)*4*PI_const*sin((a+0.5)*da)*sin(da/2);
			fprintf(pFile_outp," %E ",(double)HostMem->Rd_ra[a*nr+r]/scale2);
			if((i++)==4){i=0;fprintf(pFile_outp,"\n");}
		}
	}

	// Scale and write Tt_ra
	i=0;
	fprintf(pFile_outp,"\n\n# Tt[r][angle]. [1/(cm2sr)].\n# Tt[0][0], [0][1],..[0][na-1]\n# Tt[1][0], [1][1],..[1][na-1]\n# ...\n# Tt[nr-1][0], [nr-1][1],..[nr-1][na-1]\nTt_ra\n");
	for(r=0;r<nr;r++)
	{
		for(a=0;a<na;a++)
		{
			scale2=scale1*2*PI_const*(r+0.5)*dr*dr*cos((a+0.5)*da)*4*PI_const*sin((a+0.5)*da)*sin(da/2);
			fprintf(pFile_outp," %E ",(double)HostMem->Tt_ra[a*nr+r]/scale2);
			if((i++)==4){i=0;fprintf(pFile_outp,"\n");}
		}
	}




	fclose(pFile_outp);
	return 0;

}


int isnumeric(char a)
{
	if(a>=(char)48 && a<=(char)57) return 1;
	else return 0;
}

int readfloats(int n_floats, float* temp, FILE* pFile)
{
	int ii=0;
	char mystring [200];

	if(n_floats>NFLOATS) return 0; //cannot read more than NFLOATS floats

	while(ii<=0)
	{
		if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
		fgets(mystring , 200 , pFile);
		memset(temp,0,NFLOATS*sizeof(float));
		ii=sscanf(mystring,"%f %f %f %f %f",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
		if(ii>n_floats) return 0; //if we read more number than defined something is wrong with the file!
		//printf("ii=%d temp=%f %f %f %f %f\n",ii,temp[0],temp[1],temp[2],temp[3],temp[4]);
	}
	return 1; // Everyting appears to be ok!
}

int readints(int n_ints, int* temp, FILE* pFile) //replace with template?
{
	int ii=0;
	char mystring[STR_LEN];

	if(n_ints>NINTS) return 0; //cannot read more than NFLOATS floats

	while(ii<=0)
	{
		if(feof(pFile)) return 0; //if we reach EOF here something is wrong with the file!
		fgets(mystring , STR_LEN , pFile);
		memset(temp,0,NINTS*sizeof(int));
		ii=sscanf(mystring,"%d %d %d %d %d",&temp[0],&temp[1],&temp[2],&temp[3],&temp[4]);
		if(ii>n_ints) return 0; //if we read more number than defined something is wrong with the file!
		//printf("ii=%d temp=%f %f %f %f %f\n",ii,temp[0],temp[1],temp[2],temp[3],temp[4]);
	}
	return 1; // Everyting appears to be ok!
}

int ischar(char a)
{
	if((a>=(char)65 && a<=(char)90)||(a>=(char)97 && a<=(char)122)) return 1;
	else return 0;
}

int read_simulation_data(char* filename, SimulationStruct** simulations, int ignoreAdetection)
{
	int i=0;
	int ii=0;
	unsigned long number_of_photons;
	int n_simulations = 0;
	int n_layers = 0;
	FILE * pFile;
	char mystring [STR_LEN];
	char str[STR_LEN];
	char AorB;
	float dtot=0;


	float ftemp[NFLOATS];//Find a more elegant way to do this...
	int itemp[NINTS];

	double n1, n2, r; 

	pFile = fopen(filename , "r");
	if (pFile == NULL){perror ("Error opening file");return 0;}
	
	// First read the first data line (file version) and ignore
	if(!readfloats(1, ftemp, pFile)){perror ("Error reading file version");return 0;}
	//printf("File version: %f\n",ftemp[0]);

	// Second, read the number of runs
	if(!readints(1, itemp, pFile)){perror ("Error reading number of runs");return 0;}
	n_simulations = itemp[0];
	//printf("Number of runs: %d\n",n_simulations);
	
	// Allocate memory for the SimulationStruct array
	*simulations = (SimulationStruct*) malloc(sizeof(SimulationStruct)*n_simulations);
	if(*simulations == NULL){perror("Failed to malloc simulations.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}

	for(i=0;i<n_simulations;i++)
	{
		// Store the input filename
		strcpy((*simulations)[i].inp_filename,filename);
		// Echo the Filename
		//printf("Input filename: %s\n",filename);

		// Store ignoreAdetection data
		(*simulations)[i].ignoreAdetection=ignoreAdetection;

		// Read the output filename and determine ASCII or Binary output
		ii=0;
		while(ii<=0)
		{
			(*simulations)[i].begin=ftell(pFile);
			fgets (mystring , STR_LEN , pFile);
			ii=sscanf(mystring,"%s %c",str,&AorB);
			if(feof(pFile)|| ii>2){perror("Error reading output filename");return 0;}
			if(ii>0)ii=ischar(str[0]);
		}
		// Echo the Filename and AorB
		//printf("Output filename: %s, AorB=%c\n",str,AorB);
		strcpy((*simulations)[i].outp_filename,str);
		(*simulations)[i].AorB=AorB;

		//printf("begin=%d\n",(*simulations)[i].begin);

		// Read the number of photons
		ii=0;
		while(ii<=0)
		{
			fgets(mystring , STR_LEN , pFile);
			number_of_photons=0;
			ii=sscanf(mystring,"%lu",&number_of_photons);
			if(feof(pFile) || ii>1){perror("Error reading number of photons");return 0;} //if we reach EOF or read more number than defined something is wrong with the file!
			//printf("ii=%d temp=%f %f %f %f %f\n",ii,temp[0],temp[1],temp[2],temp[3],temp[4]);
		}
		//printf("Number of photons: %lu\n",number_of_photons);
		(*simulations)[i].number_of_photons=number_of_photons;

		// Read dr and dz (2x float)
		if(!readfloats(2, ftemp, pFile)){perror ("Error reading dr and dz");return 0;}
		//printf("dz=%f, dr=%f\n",ftemp[0],ftemp[1]);
		(*simulations)[i].det.dz=ftemp[0];
		(*simulations)[i].det.dr=ftemp[1];

		// Read No. of dz, dr and da  (3x int)
		if(!readints(3, itemp, pFile)){perror ("Error reading No. of dz, dr and da");return 0;}
		//printf("No. of dz=%d, dr=%d, da=%d\n",itemp[0],itemp[1],itemp[2]);
		(*simulations)[i].det.nz=itemp[0];
		(*simulations)[i].det.nr=itemp[1];
		(*simulations)[i].det.na=itemp[2];

		// Read No. of layers (1xint)
		if(!readints(1, itemp, pFile)){perror ("Error reading No. of layers");return 0;}
		printf("No. of layers=%d\n",itemp[0]);
		n_layers = itemp[0];
		(*simulations)[i].n_layers = itemp[0];

		// Allocate memory for the layers (including one for the upper and one for the lower)
		(*simulations)[i].layers = (LayerStruct*) malloc(sizeof(LayerStruct)*(n_layers+2));
		if((*simulations)[i].layers == NULL){perror("Failed to malloc layers.\n");return 0;}//{printf("Failed to malloc simulations.\n");return 0;}


		// Read upper refractive index (1xfloat)
		if(!readfloats(1, ftemp, pFile)){perror ("Error reading upper refractive index");return 0;}
		printf("Upper refractive index=%f\n",ftemp[0]);
		(*simulations)[i].layers[0].n=ftemp[0];

		dtot=0;
		for(ii=1;ii<=n_layers;ii++)
		{
			// Read Layer data (5x float)
			if(!readfloats(5, ftemp, pFile)){perror ("Error reading layer data");return 0;}
			printf("n=%f, mua=%f, mus=%f, g=%f, d=%f\n",ftemp[0],ftemp[1],ftemp[2],ftemp[3],ftemp[4]);
			(*simulations)[i].layers[ii].n=ftemp[0];
			(*simulations)[i].layers[ii].mua=ftemp[1];
			(*simulations)[i].layers[ii].g=ftemp[3];
			(*simulations)[i].layers[ii].z_min=dtot;
			dtot+=ftemp[4];
			(*simulations)[i].layers[ii].z_max=dtot;
			if(ftemp[2]==0.0f)(*simulations)[i].layers[ii].mutr=FLT_MAX; //Glas layer
			else(*simulations)[i].layers[ii].mutr=1.0f/(ftemp[1]+ftemp[2]);
			//printf("mutr=%f\n",(*simulations)[i].layers[ii].mutr);
			//printf("z_min=%f, z_max=%f\n",(*simulations)[i].layers[ii].z_min,(*simulations)[i].layers[ii].z_max);
		}//end ii<n_layers

		// Read lower refractive index (1xfloat)
		if(!readfloats(1, ftemp, pFile)){perror ("Error reading lower refractive index");return 0;}
		printf("Lower refractive index=%f\n",ftemp[0]);
		(*simulations)[i].layers[n_layers+1].n=ftemp[0];

		(*simulations)[i].end=ftell(pFile);
		//printf("end=%d\n",(*simulations)[i].end);

		//calculate start_weight
		n1=(*simulations)[i].layers[0].n;
		n2=(*simulations)[i].layers[1].n;
		r = (n1-n2)/(n1+n2);
		r = r*r;
		(*simulations)[i].start_weight = 1.0F - (float)r;

	}//end for i<n_simulations
	return n_simulations;
}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations)
{
	int i; 
    for(i=0;i<n_simulations;i++)free(sim[i].layers);
    free(sim);
}
