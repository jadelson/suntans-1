/*
 * File: tides.c
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * Contains functions that read/write data for specification of tidal
 * components at boundaries of type 2.
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior 
 * University. All Rights Reserved.
 *
 */
#include "suntans.h"
#include "mympi.h"
#include "grid.h"
#include "tides.h"
#include "memory.h"

static void InitTidalArrays(int N, int numtides);
static void ReadTidalArrays(FILE *ifid, char *istr, int N, int numtides);

/*
 * Function: SetTideComponents
 * Usage: SetTideComponents(grid,myproc);
 * --------------------------------------
 * This function looks for the tidal data in the file specified by
 * TideInput in suntans.dat and reads in the appropriate tidal data.
 * If that file does not exist, then the locations of the x-y points
 * of the boundaries are output to the file specified by TideOutput
 * in suntans.dat.
 *
 * Note that velocity mags must be in m/s and phases are in rad/s!!
 *
 * Format of the binary TideInput file:
 * ------------------------------------
 * numtides (1 X int) number of tidal constituents.
 * numboundaryedges (1 X int) number of boundary edges.
 * omegas (numtides X REAL) frequencies of tidal components.
 * u_amp[0] (numtides X REAL) amplitude of easting velocity at location 0.
 * u_phase[0] (numtides X REAL) phase of easting velocity at location 0.
 * v_amp[0] (numtides X REAL) amplitude of northing velocity at location 0.
 * v_phase[0] (numtides X REAL) phase of northing velocity at location 0.
 * h_amp[0] (numtides X REAL) amplitude of free surface at location 0.
 * h_phase[0] (numtides X REAL) phase of free surface at location 0.
 * u_amp[1] (numtides X REAL) amplitude of easting velocity at location 1.
 * u_phase[1] (numtides X REAL) phase of easting velocity at location 1.
 * v_amp[1] (numtides X REAL) amplitude of northing velocity at location 1.
 * v_phase[1] (numtides X REAL) phase of northing velocity at location 1.
 * h_amp[1] (numtides X REAL) amplitude of free surface at location 1.
 * h_phase[1] (numtides X REAL) phase of free surface at location 1.
 * ...
 * u_amp[numboundaryedges-1] (numtides X REAL) amplitude of easting velocity at last location.
 * u_phase[numboundaryedges-1] (numtides X REAL) phase of easting velocity at last location.
 * v_amp[numboundaryedges-1] (numtides X REAL) amplitude of northing velocity at last location.
 * v_phase[numboundaryedges-1] (numtides X REAL) phase of northing velocity at last location.
 * h_amp[numboundaryedges-1] (numtides X REAL) amplitude of free surface at last location.
 * h_phase[numboundaryedges-1] (numtides X REAL) phase of free surface at last location 
 *
 */
int SetTideComponents(gridT *grid, int myproc) {
  int j, jptr, numboundaryedges;
  char istr[BUFFERLENGTH], ostr[BUFFERLENGTH], filename[BUFFERLENGTH];
  FILE *ifid, *ofid;
  char inputbuffer[1000];

  MPI_GetFile(filename,DATAFILE,"TideInput","SetTideComponents",myproc);
  sprintf(istr,"%s.%d",filename,myproc);
  MPI_GetFile(filename,DATAFILE,"TideOutput","SetTideComponents",myproc);
  sprintf(ostr,"%s.%d",filename,myproc);
  
  if((ifid=fopen(istr,"r"))==NULL) {
    printf("Error opening %s!\n",istr);
    printf("Writing x-y boundary locations to %s instead.\n",ostr);

    ofid=fopen(ostr,"w");
    for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
      j=grid->edgep[jptr];
      
      fprintf(ofid,"%f %f\n",grid->xe[j],grid->ye[j]);
    }
    fclose(ofid);

    MPI_Finalize();
    exit(EXIT_FAILURE);
  } else {
    if(VERBOSE>2 && myproc==0) printf("Reading tidal data.\n");

    // Read in number of tidal components from file
    fgets(inputbuffer,1000,ifid);
    numtides = atoi(inputbuffer);
    // fread(&numtides,sizeof(int),1,ifid);
    // Read in number of edges in file
    fgets(inputbuffer,1000,ifid);
    numboundaryedges = atoi(inputbuffer);
    // fread(&numboundaryedges,sizeof(int),1,ifid);

    printf("num boundary edges is %d\n",numboundaryedges);
    printf("edge cell is %d and %d\n",grid->edgedist[3],grid->edgedist[2]);

    //    if(numboundaryedges!=grid->edgedist[3]-grid->edgedist[2]) {
    //  printf("Error reading %s.  Number of edges does not match current run!\n",istr);
    //  MPI_Finalize();
    // exit(EXIT_FAILURE);
    // } else {
      InitTidalArrays(numboundaryedges,numtides);
      ReadTidalArrays(ifid,istr,numboundaryedges,numtides);
      // }
  }
}

static void InitTidalArrays(int N,int numtides) {
  int j;

  // u_amp = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // v_amp = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // h_amp = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // u_phase = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // v_phase = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // h_phase = (REAL **)SunMalloc(N*sizeof(REAL *),"InitTidalArrays");
  // omegas = (REAL *)SunMalloc(numtides*sizeof(REAL),"InitTidalArrays");

  u_amp = (float **)malloc(N*sizeof(float *));
  v_amp = (float **)malloc(N*sizeof(float *));
  h_amp = (float **)malloc(N*sizeof(float *));
  u_phase = (float **)malloc(N*sizeof(float *));
  v_phase = (float **)malloc(N*sizeof(float *));
  h_phase = (float **)malloc(N*sizeof(float *));

  omegas = (float *)malloc(numtides*sizeof(float));

  /*
  real_uamp = (float **)malloc(N*sizeof(float *));
  real_vamp = (float **)malloc(N*sizeof(float *));
  real_hamp = (float **)malloc(N*sizeof(float *));
  imag_uamp = (float **)malloc(N*sizeof(float *));
  imag_vamp = (float **)malloc(N*sizeof(float *));
  imag_hamp = (float **)malloc(N*sizeof(float *));
  u_phase = (float **)malloc(N*sizeof(float *));
  v_phase = (float **)malloc(N*sizeof(float *));
  h_phase = (float **)malloc(N*sizeof(float *));
  */

  
  for(j=0;j<N;j++) {
    u_amp[j] = (float *)malloc(numtides*sizeof(float));
    v_amp[j] = (float *)malloc(numtides*sizeof(float));
    h_amp[j] = (float *)malloc(numtides*sizeof(float));
    u_phase[j] = (float *)malloc(numtides*sizeof(float));
    v_phase[j] = (float *)malloc(numtides*sizeof(float));
    h_phase[j] = (float *)malloc(numtides*sizeof(float));
  }
 
  /*
  for(j=0;j<N;j++) {
    real_uamp[j] = (float *)malloc(numtides*sizeof(float));
    real_vamp[j] = (float *)malloc(numtides*sizeof(float));
    real_hamp[j] = (float *)malloc(numtides*sizeof(float));
    imag_uamp[j] = (float *)malloc(numtides*sizeof(float));
    imag_vamp[j] = (float *)malloc(numtides*sizeof(float));
    imag_hamp[j] = (float *)malloc(numtides*sizeof(float));
    u_phase[j] = (float *)malloc(numtides*sizeof(float));
    v_phase[j] = (float *)malloc(numtides*sizeof(float));
    h_phase[j] = (float *)malloc(numtides*sizeof(float));
  }
  */
}

static void ReadTidalArrays(FILE *ifid, char *istr, int N, int numtides) {
  int t, flag=0;
  int j = 0;

  // declare array for each line of file
  char inputstring[1000];

  // read omegas
  fgets(inputstring,1000,ifid);
  sscanf(inputstring,"%f %f %f %f %f %f %f %f",&omegas[0],&omegas[1],&omegas[2],&omegas[3],&omegas[4],&omegas[5],&omegas[6],&omegas[7]);
  printf("omega is %f\n",omegas[0]);

  // read u,v,h
  while(fgets(inputstring,1000,ifid)!=NULL) {
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&u_amp[j][0],&u_amp[j][1],&u_amp[j][2],&u_amp[j][3],&u_amp[j][4],&u_amp[j][5],&u_amp[j][6],&u_amp[j][7]);   
    printf("j is %d\n",j);
    printf("u amp is %f\n",u_amp[j][0]);

    fgets(inputstring,1000,ifid);          
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&u_phase[j][0],&u_phase[j][1],&u_phase[j][2],&u_phase[j][3],&u_phase[j][4],&u_phase[j][5],&u_phase[j][6],&u_phase[j][7]);  
    printf("u phase is %f\n",u_phase[j][0]);

    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&v_amp[j][0],&v_amp[j][1],&v_amp[j][2],&v_amp[j][3],&v_amp[j][4],&v_amp[j][5],&v_amp[j][6],&v_amp[j][7]);
    printf("v amp is %f\n",v_amp[j][0]);

    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&v_phase[j][0],&v_phase[j][1],&v_phase[j][2],&v_phase[j][3],&v_phase[j][4],&v_phase[j][5],&v_phase[j][6],&v_phase[j][7]);
    printf("v phase is %f\n",v_phase[j][0]);

    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&h_amp[j][0],&h_amp[j][1],&h_amp[j][2],&h_amp[j][3],&h_amp[j][4],&h_amp[j][5],&h_amp[j][6],&h_amp[j][7]);
    printf("h amp is %f\n",h_amp[j][0]);

    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&h_phase[j][0],&h_phase[j][1],&h_phase[j][2],&h_phase[j][3],&h_phase[j][4],&h_phase[j][5],&h_phase[j][6],&h_phase[j][7]);
    printf("h phase is %f\n",h_phase[j][0]);


    /*
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&real_uamp[j][0],&real_uamp[j][1],&real_uamp[j][2],&real_uamp[j][3],&real_uamp[j][4],&real_uamp[j][5],&real_uamp[j][6],&real_uamp[j][7]);
    printf("real uamp is %f\n",real_uamp[j][0]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&imag_uamp[j][0],&imag_uamp[j][1],&imag_uamp[j][2],&imag_uamp[j][3],&imag_uamp[j][4],&imag_uamp[j][5],&imag_uamp[j][6],&imag_uamp[j][7]);
    printf("imag uamp is %f\n",imag_uamp[j][0]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&u_phase[j][0],&u_phase[j][1],&u_phase[j][2],&u_phase[j][3],&u_phase[j][4],&u_phase[j][5],&u_phase[j][6],&u_phase[j][7]);
    printf("u phase is %f\n",u_phase[j][0]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&real_vamp[j][0],&real_vamp[j][1],&real_vamp[j][2],&real_vamp[j][3],&real_vamp[j][4],&real_vamp[j][5],&real_vamp[j][6],&real_vamp[j][7]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&imag_vamp[j][0],&imag_vamp[j][1],&imag_vamp[j][2],&imag_vamp[j][3],&imag_vamp[j][4],&imag_vamp[j][5],&imag_vamp[j][6],&imag_vamp[j][7]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&v_phase[j][0],&v_phase[j][1],&v_phase[j][2],&v_phase[j][3],&v_phase[j][4],&v_phase[j][5],&v_phase[j][6],&v_phase[j][7]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&real_hamp[j][0],&real_hamp[j][1],&real_hamp[j][2],&real_hamp[j][3],&real_hamp[j][4],&real_hamp[j][5],&real_hamp[j][6],&real_hamp[j][7]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&imag_hamp[j][0],&imag_hamp[j][1],&imag_hamp[j][2],&imag_hamp[j][3],&imag_hamp[j][4],&imag_hamp[j][5],&imag_hamp[j][6],&imag_hamp[j][7]);
    fgets(inputstring,1000,ifid);
    sscanf(inputstring,"%f %f %f %f %f %f %f %f",&h_phase[j][0],&h_phase[j][1],&h_phase[j][2],&h_phase[j][3],&h_phase[j][4],&h_phase[j][5],&h_phase[j][6],&h_phase[j][7]);
    */
    j++;
  
  //  flag=1;
  }
  
  if(flag)
    printf("Error reading tidal data.  Only %d of %d edges in %s.\n",j,N,istr);
}
    
  

  
    
