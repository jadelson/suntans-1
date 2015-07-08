/*                                                                                           
 * File: boundaries.c                                                                        
 * Author: Oliver B. Fringer                                                                 
 * Institution: Stanford University                                                          
 * --------------------------------                                                          
 * This file contains functions to impose the boundary conditions on u.                      
 *                                                                                           
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior               
 * University. All Rights Reserved.                                                          
 * 
 */
#include "boundaries.h"
#include <math.h>

/*
 * Function: OpenBoundaryFluxes
 * Usage: OpenBoundaryFluxes(q,ubnew,ubn,grid,phys,prop);
 * ----------------------------------------------------
 * This will update the boundary flux at the edgedist[2] to edgedist[3] edges.
 * 
 * Note that phys->uold,vold contain the velocity at time step n-1 and 
 * phys->uc,vc contain it at time step n.
 *
 * The radiative open boundary condition does not work yet!!!  For this reason c[k] is
 * set to 0
 *
 */
void OpenBoundaryFluxes(REAL **q, REAL **ub, REAL **ubn, gridT *grid, physT *phys, propT *prop) {
  int j, jptr, ib, k, forced;
  REAL *uboundary = phys->a, **u = phys->uc, **v = phys->vc, **uold = phys->uold, **vold = phys->vold;
  REAL z, c0, c1, C0, C1, dt=prop->dt, u0, u0new, uc0, vc0, uc0old, vc0old, ub0;

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];

    ib = grid->grad[2*j];

    for(k=grid->etop[j];k<grid->Nke[j];k++) 
      ub[j][k]=phys->boundary_u[jptr-grid->edgedist[2]][k]*grid->n1[j]+
	phys->boundary_v[jptr-grid->edgedist[2]][k]*grid->n2[j];
  }
}


/*
 * Function: BoundaryScalars
 * Usage: BoundaryScalars(boundary_s,boundary_T,grid,phys,prop);
 * -------------------------------------------------------------
 * This will set the values of the scalars at the open boundaries.
 * 
 */
void BoundaryScalars(gridT *grid, physT *phys, propT *prop) {
 int jptr, j, ib, k, iptr, i;
  REAL z;
  
  // At the ocean boundary
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];

    for(k=grid->ctop[i];k<grid->Nk[i];k++) {
      phys->s[i][k]=32;
      phys->T[i][k]=0;
    }
  }

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];
    ib = grid->grad[2*j];

    for(k=grid->ctop[ib];k<grid->Nk[ib];k++) {
      phys->boundary_T[jptr-grid->edgedist[2]][k]=0;
      phys->boundary_s[jptr-grid->edgedist[2]][k]=0;
    }
  }

}

/*
 * Function: BoundaryVelocities 
 * Usage: BoundaryVelocities(grid,phys,prop);
 * ------------------------------------------
 * This will set the values of u,v,w, and h at the boundaries.
 */
void BoundaryVelocities1(gridT *grid, physT *phys, propT *prop, int myproc) {
  // void BoundaryVelocities(gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm) {
  int jptr, j, ib, k, iptr, i, index, n, numtides = 8, days = 30;
  REAL z, cb, h, u0=0.1, toffSet, secondsPerDay = 86400.0;
  REAL flow_sac, flow_sj, area_sac, area_sj;
  REAL *omegas,*h_amp,*h_phase;
  REAL fc1, fc2, tmp;
  int Ninput1, Ninput2, Nmax;
  

  Nmax = prop->nsteps*prop->dt/prop->dt_tideBC;
  Ninput1 = (int)(prop->rtime/prop->dt_tideBC);
  Ninput2 = Ninput1 + 1;

  if (Ninput1 >= Nmax-1){
    Ninput1 = Nmax-1;
    Ninput2 = Ninput1;
  }

  fc1 = prop->rtime - (double)(Ninput1*prop->dt_tideBC);
  fc1 /= prop->dt_tideBC; 
  fc2 = 1.0-fc1;
  tmp = phys->wl_tideBC[Ninput1]*fc2 + phys->wl_tideBC[Ninput2]*fc1;

  if(myproc == 0){
    printf("Tide BC at Ponit Reyes is %f m\n ", tmp);
  }

  toffSet = MPI_GetValue(DATAFILE,"toffSet","BoundaryVelocities",myproc)*secondsPerDay;
  
  omegas = malloc(numtides*sizeof(REAL));
  h_amp = malloc(numtides*sizeof(REAL));
  h_phase = malloc(numtides*sizeof(REAL));

  omegas[0] = 0.0001405189;
  omegas[1] = 0.0001454441;
  omegas[2] = 0.0001378796;
  omegas[3] = 0.0001458423;
  omegas[4] = 0.0000675977;
  omegas[5] = 0.0000729211;
  omegas[6] = 0.0000725229;
  omegas[7] = 0.0000649585;

  h_amp[0] = 0.559758;
  h_amp[1] = 0.137000;
  h_amp[2] = 0.118707;
  h_amp[3] = 0.052000;
  h_amp[4] = 0.270342;
  h_amp[5] = 0.407854;
  h_amp[6] = 0.116000;
  h_amp[7] = 0.047016;

  h_phase[0] = 0.352032;
  h_phase[1] = -3.811796;
  h_phase[2] = 2.926566;
  h_phase[3] = -0.238063;
  h_phase[4] = 0.233525;
  h_phase[5] = -3.808654;
  h_phase[6] = 2.198240;
  h_phase[7] = 2.479937;



  // At the ocean boundary
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
    h = 0;
    for(n=0;n<numtides;n++) {
      h = h + h_amp[n]*cos(omegas[n]*(toffSet+prop->rtime) + h_phase[n]);
    }
    phys->h[i] = h;
  }
  
  flow_sac = 300;
  flow_sj = 300;
  
  area_sac = 0;
  area_sj = 0;
  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];
    ib = grid->grad[2*j];
    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      if(grid->yv[ib]>4215000.0)
        area_sac = area_sac + grid->dzz[ib][k]*grid->df[j];
      else
        area_sj = area_sj + grid->dzz[ib][k]*grid->df[j];
    }
  }

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];
    ib = grid->grad[2*j];

    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      if(grid->yv[ib]>4215000.0) {
	phys->boundary_u[jptr-grid->edgedist[2]][k] = -flow_sac/area_sac;
      }
      else {
	phys->boundary_u[jptr-grid->edgedist[2]][k] = -flow_sj/area_sj;
      }
      phys->boundary_v[jptr-grid->edgedist[2]][k]=0.0;
      phys->boundary_w[jptr-grid->edgedist[2]][k]=0.0;
    }
  }

}

void BoundaryVelocities(gridT *grid, physT *phys, propT *prop, int myproc) {
  // void BoundaryVelocities(gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm) {
  int jptr, j, ib, k, iptr, i, index, n, numtides = 8, days = 31, rday;
  REAL z, cb, h, u0=0.1, toffSet, secondsPerDay = 86400.0;
  REAL area_sac, area_sj;
  REAL fc1, fc2, tmp;
  REAL ftc2mtrc, rtime_rel;
  REAL *flow_sac, *flow_sj;
  int Ninput1, Ninput2, Nmax;
  
  rday = (int)floor(prop->rtime/secondsPerDay);
  if (rday > days) rday = days;  
  flow_sac = malloc(days*sizeof(REAL));
  flow_sj  = malloc(days*sizeof(REAL));

  Nmax = prop->nsteps*prop->dt/prop->dt_tideBC;
  rtime_rel = (prop->n - prop->nstart)*prop->dt;
  Ninput1 = (int)(rtime_rel/prop->dt_tideBC);
  Ninput2 = Ninput1 + 1;

  if (Ninput1 >= Nmax-1){
    Ninput1 = Nmax-1;
    Ninput2 = Ninput1;
  }

  fc1 = rtime_rel - (double)(Ninput1*prop->dt_tideBC);
  fc1 /= prop->dt_tideBC; 
  fc2 = 1.0-fc1;
  tmp = phys->wl_tideBC[Ninput1]*fc2 + phys->wl_tideBC[Ninput2]*fc1;

  if(myproc == 0){
    printf("Tide BC at Ponit Reyes is %f m\n ", tmp);
  }

  toffSet = MPI_GetValue(DATAFILE,"toffSet","BoundaryVelocities",myproc)*secondsPerDay;


  // At the ocean boundary
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
    phys->h[i] = tmp;
  }
  
  ftc2mtrc = pow(0.3048,3);
  flow_sac[0] = 15900*ftc2mtrc; 
  flow_sac[1] = 15200*ftc2mtrc; 
  flow_sac[2] = 15100*ftc2mtrc; 
  flow_sac[3] = 14800*ftc2mtrc; 
  flow_sac[4] = 14700*ftc2mtrc; 
  flow_sac[5] = 15200*ftc2mtrc; 
  flow_sac[6] = 17000*ftc2mtrc; 
  flow_sac[7] = 19900*ftc2mtrc; 
  flow_sac[8] = 21800*ftc2mtrc; 
  flow_sac[9] = 26200*ftc2mtrc; 
  flow_sac[10] = 35200*ftc2mtrc; 
  flow_sac[11]= 38600*ftc2mtrc; 
  flow_sac[12] = 41100*ftc2mtrc; 
  flow_sac[13] = 41400*ftc2mtrc; 
  flow_sac[14] = 41100*ftc2mtrc; 
  flow_sac[15] = 40200*ftc2mtrc; 
  flow_sac[16] = 40600*ftc2mtrc; 
  flow_sac[17] = 44200*ftc2mtrc; 
  flow_sac[18] = 48000*ftc2mtrc; 
  flow_sac[19] = 66900*ftc2mtrc; 
  flow_sac[20] = 72300*ftc2mtrc; 
  flow_sac[21] = 74100*ftc2mtrc; 
  flow_sac[22] = 70800*ftc2mtrc; 
  flow_sac[23] = 68200*ftc2mtrc; 
  flow_sac[24] = 64500*ftc2mtrc; 
  flow_sac[25] = 59500*ftc2mtrc; 
  flow_sac[26] = 53200*ftc2mtrc; 
  flow_sac[27] = 46800*ftc2mtrc; 
  flow_sac[28] = 42100*ftc2mtrc; 
  flow_sac[29] = 40500*ftc2mtrc; 
  flow_sac[30] = 41700*ftc2mtrc; 

  flow_sj[0] = 7720*ftc2mtrc; 
  flow_sj[1] = 8180*ftc2mtrc; 
  flow_sj[2] = 8320*ftc2mtrc; 
  flow_sj[3] = 8070*ftc2mtrc; 
  flow_sj[4] = 7890*ftc2mtrc; 
  flow_sj[5] = 8130*ftc2mtrc; 
  flow_sj[6] = 8400*ftc2mtrc; 
  flow_sj[7] = 8610*ftc2mtrc; 
  flow_sj[8] = 8820*ftc2mtrc; 
  flow_sj[9] = 9060*ftc2mtrc; 
  flow_sj[10] = 9110*ftc2mtrc; 
  flow_sj[11] = 9070*ftc2mtrc; 
  flow_sj[12] = 9130*ftc2mtrc; 
  flow_sj[13] = 9220*ftc2mtrc; 
  flow_sj[14] = 9250*ftc2mtrc; 
  flow_sj[15] = 9120*ftc2mtrc; 
  flow_sj[16] = 8970*ftc2mtrc; 
  flow_sj[17] = 8940*ftc2mtrc; 
  flow_sj[18] = 9340*ftc2mtrc; 
  flow_sj[19] = 10200*ftc2mtrc; 
  flow_sj[20] = 11400*ftc2mtrc; 
  flow_sj[21] = 12100*ftc2mtrc; 
  flow_sj[22] = 12600*ftc2mtrc; 
  flow_sj[23] = 13000*ftc2mtrc; 
  flow_sj[24] = 13200*ftc2mtrc; 
  flow_sj[25] = 13500*ftc2mtrc; 
  flow_sj[26] = 13500*ftc2mtrc; 
  flow_sj[27] = 13800*ftc2mtrc; 
  flow_sj[28] = 14200*ftc2mtrc; 
  flow_sj[29] = 14700*ftc2mtrc; 
  flow_sj[30] = 15100*ftc2mtrc; 

 
  area_sac = 0;
  area_sj = 0;
  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];
    ib = grid->grad[2*j];
    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      if(grid->yv[ib]>4215000.0)
        area_sac = area_sac + grid->dzz[ib][k]*grid->df[j];
      else
        area_sj = area_sj + grid->dzz[ib][k]*grid->df[j];
    }
  }

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];
    ib = grid->grad[2*j];

    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      if(grid->yv[ib]>4215000.0) {
	//	phys->boundary_u[jptr-grid->edgedist[2]][k] = -flow_sac[rday]/area_sac;
	phys->boundary_u[jptr-grid->edgedist[2]][k] = -300/area_sac;
      }
      else {
	//	phys->boundary_u[jptr-grid->edgedist[2]][k] = -flow_sj[rday]/area_sj;
	phys->boundary_u[jptr-grid->edgedist[2]][k] = -300/area_sj;
      }
      phys->boundary_v[jptr-grid->edgedist[2]][k]=0.0;
      phys->boundary_w[jptr-grid->edgedist[2]][k]=0.0;
    }
  }

}


/*
 * Function: WindStress
 * Usage: WindStress(grid,phys,prop);
 * ----------------------------------
 * Set the wind stress as well as the bottom stress.
 * tau_B is not currently in use (4/1/05).
 *
 */
void WindStress(gridT *grid, physT *phys, propT *prop) {
  int j, jptr;

  for(jptr=grid->edgedist[0];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    phys->tau_T[j]=grid->n2[j]*prop->tau_Tp;
    phys->tau_B[j]=0;
  }
}
