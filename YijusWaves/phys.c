/*
 * File: phys.c
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * This file contains physically-based functions.
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior
 * University. All Rights Reserved.
 *
 */
#include "suntans.h"
#include "phys.h"
#include "grid.h"
#include "util.h"
#include "initialization.h"
#include "memory.h"
#include "turbulence.h"
#include "boundaries.h"
#include "profiles.h"
#include "state.h"
#include "scalars.h"
#include "sedi.h"
#include "wave.h"

/*
 * Private Function declarations.
 *
 */
static void UpdateDZ(gridT *grid, physT *phys, int option);
static void UPredictor(gridT *grid, physT *phys,  
		       propT *prop, int myproc, int numprocs, MPI_Comm comm);
static void Corrector(REAL **qc, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm);
static void ComputeQSource(REAL **src, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs);
static void CGSolve(gridT *grid, physT *phys, propT *prop, 
		    int myproc, int numprocs, MPI_Comm comm);
static void CGSolveQ(REAL **q, REAL **src, REAL **c, gridT *grid, physT *phys, propT *prop, 
		     int myproc, int numprocs, MPI_Comm comm);
static void ConditionQ(REAL **x, gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm);
static void Preconditioner(REAL **x, REAL **xc, REAL **coef, gridT *grid, physT *phys, propT *prop);
static void GuessQ(REAL **q, REAL **wold, REAL **w, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm);
static void GSSolve(gridT *grid, physT *phys, propT *prop, 
		    int myproc, int numprocs, MPI_Comm comm);
static REAL InnerProduct(REAL *x, REAL *y, gridT *grid, int myproc, int numprocs, MPI_Comm comm);
static REAL InnerProduct3(REAL **x, REAL **y, gridT *grid, int myproc, int numprocs, MPI_Comm comm);
static void OperatorH(REAL *x, REAL *y, gridT *grid, physT *phys, propT *prop);
static void OperatorQC(REAL **coef, REAL **fcoef, REAL **x, REAL **y, REAL **c, gridT *grid, physT *phys, propT *prop);
static void QCoefficients(REAL **coef, REAL **fcoef, REAL **c, gridT *grid, physT *phys, propT *prop);
static void OperatorQ(REAL **coef, REAL **x, REAL **y, REAL **c, gridT *grid, physT *phys, propT *prop);
static void Continuity(REAL **w, gridT *grid, physT *phys, propT *prop);
static void ComputeConservatives(gridT *grid, physT *phys, propT *prop, int myproc, int numprocs,
			  MPI_Comm comm);
static int Check(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave, wpropT *wprop, 
		 int myproc, int numprocs, MPI_Comm comm);
static void Progress(propT *prop, int myproc);
static void EddyViscosity(gridT *grid, physT *phys, sediT *sedi, propT *prop, MPI_Comm comm, int myproc);
static void HorizontalSource(gridT *grid, physT *phys, propT *prop, waveT *wave, wpropT *wprop,
			     int myproc, int numprocs, MPI_Comm comm);
static void StoreVariables(gridT *grid, physT *phys);
static void NewCells(gridT *grid, physT *phys, propT *prop);
static void WPredictor(gridT *grid, physT *phys, propT *prop,
		       int myproc, int numprocs, MPI_Comm comm);
static void ComputeVelocityVector(physT *phys, REAL **uc, REAL **vc, gridT *grid);
static void OutputData(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave, wpropT *wprop,
		int myproc, int numprocs, int blowup, MPI_Comm comm);
static REAL InterpToFace(int j, int k, REAL **phi, REAL **u, gridT *grid);
static void SetDensity(gridT *grid, physT *phys, sediT *sedi, propT *prop, spropT *sprop);
//Added for Lagrangian tracing
static void LagraTracing(gridT *grid, physT *phys, propT *prop, int start_cell, int start_layer, int *ii, int *kk, REAL *xs, REAL *zs,int myproc);
static void InterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us, REAL *al, REAL *vtx, REAL *vty,int ifstdout);
static void StableInterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us);

/*
 * Function: AllocatePhysicalVariables
 * Usage: AllocatePhysicalVariables(grid,phys,prop);
 * -------------------------------------------------
 * This function allocates space for the physical arrays but does not
 * allocate space for the grid as this has already been allocated.
 *
 */
void AllocatePhysicalVariables(gridT *grid, physT **phys, propT *prop)
{
  int flag=0, i, j, jptr, ib, Nc=grid->Nc, Ne=grid->Ne, nf, a;

  *phys = (physT *)SunMalloc(sizeof(physT),"AllocatePhysicalVariables");

  (*phys)->u = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->uc = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->vc = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->tau_SD = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->uold = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->vold = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->D = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->utmp = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->utmp2 = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->ut = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->Cn_U = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");

  (*phys)->wf = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");

  for(j=0;j<Ne;j++) {
    if(grid->Nkc[j]<grid->Nke[j]) {
      printf("Error!  Nkc(=%d)<Nke(=%d) at edge %d\n",grid->Nkc[j],grid->Nke[j],j);
      flag = 1;
    }
    (*phys)->u[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->utmp[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->utmp2[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->ut[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->Cn_U[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->wf[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
  }
  if(flag) {
    MPI_Finalize();
    exit(0);
  }

  (*phys)->h = (REAL *)SunMalloc(Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->hold = (REAL *)SunMalloc(Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->dhdt = (REAL *)SunMalloc(Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->htmp = (REAL *)SunMalloc(Nc*sizeof(REAL),"AllocatePhysicalVariables");
  
  (*phys)->w = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->wtmp = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->wtmp2 = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->Cn_W = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->q = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->qc = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->qtmp = (REAL **)SunMalloc(NFACES*Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->s = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->T = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->s0 = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->rho = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->Cn_R = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->Cn_T = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->stmp = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->stmp2 = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->stmp3 = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->nu_tv = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->kappa_tv = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->wl_tideBC = (REAL *)SunMalloc(prop->nsteps*prop->dt/prop->dt_tideBC*sizeof(REAL),"AllocatePhysicalVariables");
  if(prop->turbmodel) {
    (*phys)->qT = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
    (*phys)->lT = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
    (*phys)->Cn_q = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
    (*phys)->Cn_l = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  }
  (*phys)->tau_T = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->tau_B = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->CdT = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->CdB = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->de = (REAL *)SunMalloc(Ne*sizeof(REAL),"AllocatePhysicalVariables");

  for(i=0;i<Nc;i++) {
    (*phys)->uc[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->vc[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->tau_SD[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->uold[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->vold[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->w[i] = (REAL *)SunMalloc((grid->Nk[i]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->wtmp[i] = (REAL *)SunMalloc((grid->Nk[i]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->wtmp2[i] = (REAL *)SunMalloc((grid->Nk[i]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->Cn_W[i] = (REAL *)SunMalloc((grid->Nk[i]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->q[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->qc[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    for(nf=0;nf<NFACES;nf++)
      (*phys)->qtmp[i*NFACES+nf] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->s[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->T[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->s0[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->rho[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->Cn_R[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->Cn_T[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    if(prop->turbmodel) {
      (*phys)->Cn_q[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
      (*phys)->Cn_l[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
      (*phys)->qT[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
      (*phys)->lT[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    }
    (*phys)->stmp[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->stmp2[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->stmp3[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->nu_tv[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->kappa_tv[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
  }

  (*phys)->boundary_u = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_v = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_w = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_s = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_T = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_rho = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_tmp = (REAL **)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->boundary_h = (REAL *)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->boundary_flag = (REAL *)SunMalloc((grid->edgedist[5]-grid->edgedist[2])*sizeof(REAL),"AllocatePhysicalVariables");
  for(jptr=grid->edgedist[2];jptr<grid->edgedist[5];jptr++) {
    j=grid->edgep[jptr];

    (*phys)->boundary_u[jptr-grid->edgedist[2]] = (REAL *)SunMalloc(grid->Nke[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_v[jptr-grid->edgedist[2]] = (REAL *)SunMalloc(grid->Nke[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_w[jptr-grid->edgedist[2]] = (REAL *)SunMalloc((grid->Nke[j]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_s[jptr-grid->edgedist[2]] = (REAL *)SunMalloc(grid->Nke[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_T[jptr-grid->edgedist[2]] = (REAL *)SunMalloc(grid->Nke[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_tmp[jptr-grid->edgedist[2]] = (REAL *)SunMalloc((grid->Nke[j]+1)*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->boundary_rho[jptr-grid->edgedist[2]] = (REAL *)SunMalloc(grid->Nke[j]*sizeof(REAL),"AllocatePhysicalVariables");
 
  }

  (*phys)->ap = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->am = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->bp = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->bm = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->a = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->b = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->c = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->d = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");

  // Allocate for the face scalar
  (*phys)->SfHp = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->SfHm = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  for(j=0;j<Ne;j++) {
    (*phys)->SfHp[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->SfHm[j] = (REAL *)SunMalloc(grid->Nkc[j]*sizeof(REAL),"AllocatePhysicalVariables");
  }

  // Allocate for TVD schemes
  (*phys)->Cp = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->Cm = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->rp = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->rm = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");

  (*phys)->wp = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->wm = (REAL *)SunMalloc((grid->Nkmax+1)*sizeof(REAL),"AllocatePhysicalVariables");

  (*phys)->gradSx = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->gradSy = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  for(i=0;i<Nc;i++) {
    (*phys)->gradSx[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->gradSy[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
  }

  (*phys)->rterm = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->rterm2 = (REAL **)SunMalloc(Ne*sizeof(REAL *),"AllocatePhysicalVariables");
  for(i=0;i<Ne;i++) {
    (*phys)->rterm[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->rterm2[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
  }
  
  (*phys)->udfsminus = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->udfminus = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  for(i=0;i<Nc;i++) {
    (*phys)->udfsminus[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->udfminus[i] = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"AllocatePhysicalVariables");
  }
  
  (*phys)->udfs = (REAL *)SunMalloc(NFACES*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->udf = (REAL *)SunMalloc(NFACES*sizeof(REAL),"AllocatePhysicalVariables");

  (*phys)->ustvd = (REAL *)SunMalloc(NFACES*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->ustvd_minus = (REAL *)SunMalloc(NFACES*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->ustvd_plus = (REAL *)SunMalloc(NFACES*sizeof(REAL),"AllocatePhysicalVariables");

  (*phys)->tvdminus = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  (*phys)->tvdplus = (REAL **)SunMalloc(Nc*sizeof(REAL *),"AllocatePhysicalVariables");
  for(i=0;i<Nc;i++) {
    (*phys)->tvdminus[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
    (*phys)->tvdplus[i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL),"AllocatePhysicalVariables");
  }

  // read salinity
  (*phys)->xsal = (REAL *)SunMalloc(10*grid->Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->ysal = (REAL *)SunMalloc(10*grid->Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->zsal = (REAL *)SunMalloc(10*grid->Nc*sizeof(REAL),"AllocatePhysicalVariables");
  (*phys)->sal = (REAL *)SunMalloc(10*grid->Nc*sizeof(REAL),"AllocatePhysicalVariables");

}

/*
 * Function: FreePhysicalVariables
 * Usage: FreePhysicalVariables(grid,phys,prop);
 * ---------------------------------------------
 * This function frees all space allocated in AllocatePhysicalVariables
 *
 */
void FreePhysicalVariables(gridT *grid, physT *phys, propT *prop)
{
  int i, j, Nc=grid->Nc, Ne=grid->Ne, nf;
  
  for(j=0;j<Ne;j++) {
    free(phys->u[j]);
    free(phys->utmp[j]);
    free(phys->utmp2[j]);
    free(phys->ut[j]);
    free(phys->Cn_U[j]);
    free(phys->wf[j]);
  }

  for(i=0;i<Nc;i++) {
    free(phys->uc[i]);
    free(phys->vc[i]);
    free(phys->tau_SD[i]);
    free(phys->uold[i]);
    free(phys->vold[i]);
    free(phys->w[i]);
    free(phys->wtmp[i]);
    free(phys->wtmp2[i]);
    free(phys->Cn_W[i]);
    free(phys->q[i]);
    free(phys->qc[i]);
    for(nf=0;nf<NFACES;nf++)
      free(phys->qtmp[i*NFACES+nf]);
    free(phys->s[i]);
    free(phys->T[i]);
    free(phys->s0[i]);
    free(phys->rho[i]);
    free(phys->Cn_R[i]);
    free(phys->Cn_T[i]);
    if(prop->turbmodel) {
      free(phys->Cn_q[i]);
      free(phys->Cn_l[i]);
      free(phys->qT[i]);
      free(phys->lT[i]);
    }
    free(phys->stmp[i]);
    free(phys->stmp2[i]);
    free(phys->stmp3[i]);
    free(phys->nu_tv[i]);
    free(phys->kappa_tv[i]);
  }

  free(phys->h);
  free(phys->dhdt);
  free(phys->htmp);
  free(phys->uc);
  free(phys->vc);
  free(phys->tau_SD);
  free(phys->w);
  free(phys->wtmp);
  free(phys->wtmp2);
  free(phys->Cn_W);
  free(phys->wf);
  free(phys->q);
  free(phys->qtmp);
  free(phys->s);
  free(phys->T);
  free(phys->s0);
  free(phys->Cn_R);
  free(phys->Cn_T);
  if(prop->turbmodel) {
    free(phys->Cn_q);
    free(phys->Cn_l);
    free(phys->qT);
    free(phys->lT);
  }  
  free(phys->stmp);
  free(phys->stmp2);
  free(phys->stmp3);
  free(phys->nu_tv);
  free(phys->kappa_tv);
  free(phys->tau_T);
  free(phys->tau_B);
  free(phys->CdT);
  free(phys->CdB);
  free(phys->de);
  free(phys->u);
  free(phys->D);
  free(phys->utmp);
  free(phys->ut);
  free(phys->Cn_U);

  free(phys->ap);
  free(phys->am);
  free(phys->bp);
  free(phys->bm);
  free(phys->a);
  free(phys->b);
  free(phys->c);
  free(phys->d);

  // Free the horizontal facial scalar
  for(j=0;j<Ne;j++) {
    free( phys->SfHp[j] );
    free( phys->SfHm[j] );
  }
  free(phys->SfHp);
  free(phys->SfHm);

  // Free the variables for TVD scheme
  free(phys->Cp);
  free(phys->Cm);
  free(phys->rp);
  free(phys->rm);
  free(phys->wp);
  free(phys->wm);

  free(phys);
}
    
/*
 * Function: ReadPhysicalVariables
 * Usage: ReadPhysicalVariables(grid,phys,prop,myproc);
 * ----------------------------------------------------
 * This function reads in physical variables for a restart run
 * from the restart file defined by prop->StartFID.
 *
 */
void ReadPhysicalVariables(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, 
			   waveT *wave, wpropT *wprop, int myproc, MPI_Comm comm) {

  int i, j;
  int m, n, s, l;

  if(VERBOSE>1 && myproc==0) printf("Reading from rstore...\n");
    
  fread(&(prop->nstart),sizeof(int),1,prop->StartFID);

  fread(phys->h,sizeof(REAL),grid->Nc,prop->StartFID);
  for(j=0;j<grid->Ne;j++) 
    fread(phys->Cn_U[j],sizeof(REAL),grid->Nke[j],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->Cn_W[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->Cn_R[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->Cn_T[i],sizeof(REAL),grid->Nk[i],prop->StartFID);

  if(prop->turbmodel) {
    for(i=0;i<grid->Nc;i++) 
      fread(phys->Cn_q[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
    for(i=0;i<grid->Nc;i++) 
      fread(phys->Cn_l[i],sizeof(REAL),grid->Nk[i],prop->StartFID);

    for(i=0;i<grid->Nc;i++) 
      fread(phys->qT[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
    for(i=0;i<grid->Nc;i++) 
      fread(phys->lT[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  }
  for(i=0;i<grid->Nc;i++) 
    fread(phys->nu_tv[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->kappa_tv[i],sizeof(REAL),grid->Nk[i],prop->StartFID);

  for(j=0;j<grid->Ne;j++) 
    fread(phys->u[j],sizeof(REAL),grid->Nke[j],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->w[i],sizeof(REAL),grid->Nk[i]+1,prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->q[i],sizeof(REAL),grid->Nk[i],prop->StartFID);

  for(i=0;i<grid->Nc;i++) 
    fread(phys->s[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->T[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
  for(i=0;i<grid->Nc;i++) 
    fread(phys->s0[i],sizeof(REAL),grid->Nk[i],prop->StartFID);

  if (prop->wave)
    for(m=0; m<wprop->Mw; m++)
      for(n=0; n<wprop->Nw; n++)
	fread(wave->N[m][n],sizeof(REAL),grid->Nc,prop->StartFID);

  if (prop->sedi){
    for(s=0; s<sprop->Nsize; s++)
      for(i=0; i<grid->Nc; i++)
	fread(sedi->sd[s][i],sizeof(REAL),grid->Nk[i],prop->StartFID);

    for(i=0; i<grid->Nc; i++){
      fread(sedi->sdtot[i],sizeof(REAL),grid->Nk[i],prop->StartFID);
      fread(sedi->M[i],sizeof(REAL),sprop->NL,prop->StartFID);
    }
      
  }
  fclose(prop->StartFID);

  UpdateDZ(grid,phys,1);
  ComputeVelocityVector(phys,phys->uc,phys->vc,grid);

  // apr20 bing
  ISendRecvCellData3D(phys->uc,grid,myproc,comm);
  ISendRecvCellData3D(phys->vc,grid,myproc,comm);
}

/*
 * Function: InitializePhyiscalVariables
 * Usage: InitializePhyiscalVariables(grid,phys,prop);
 * ---------------------------------------------------
 * This function initializes the physical variables by calling
 * the routines defined in the file initialize.c
 *
 */
void InitializePhysicalVariables(gridT *grid, physT *phys, sediT *sedi, propT *prop, spropT *sprop)
{
  int i, j, k, Nc=grid->Nc;
  REAL z, *stmp;

  prop->nstart=0;

  // Initialize the free surface
  for(i=0;i<Nc;i++) {
    phys->h[i]=ReturnFreeSurface(grid->xv[i],grid->yv[i],grid->dv[i]);
    phys->dhdt[i] = 0;
    if(phys->h[i]<-grid->dv[i])
      phys->h[i]=-grid->dv[i]; //+ 1e-10*grid->dz[grid->Nk[i]-1];
  }

  // Need to update the vertical grid after updating the free surface.
  // The 1 indicates that this is the first call to UpdateDZ
  UpdateDZ(grid,phys,1);

  for(i=0;i<Nc;i++) {
    phys->w[i][grid->Nk[i]]=0;
    for(k=0;k<grid->Nk[i];k++) {
      phys->w[i][k]=0;
      phys->q[i][k]=0;
      phys->s[i][k]=0;
      phys->T[i][k]=0;
      phys->s0[i][k]=0;
      phys->qc[i][k]=0;
      phys->tau_SD[i][k]=0;
    }
  }

  // Initialize the temperature, salinity, and background salinity
  // distributions.  Since z is not stored, need to use dz[k] to get
  // z[k].
  if(prop->readSalinity) {
    // Vivien Sept 2008
    char inputstring[1000];
    float xnum, ynum, znum, salnum;
    REAL x_p1, y_p1, z_p1, x_p2, y_p2, z_p2, x_p3, y_p3, z_p3, x_p4, y_p4, z_p4;
    REAL s_p1, s_p2, s_p3, s_p4;
    REAL s_inp;
    REAL w1, w2, w3, w4;
    REAL dist, dist_p1, dist_p2, dist_p3, dist_p4;

    int index = 0;
    while(fgets(inputstring,1000,prop->InitSalinityFID)) {
      sscanf(inputstring,"%f %f %f %f",&xnum,&ynum,&znum,&salnum);
      phys->xsal[index] = xnum;
      phys->ysal[index] = ynum;
      phys->zsal[index] = znum;
      phys->sal[index] = salnum;
      index++;
    }
    fclose(prop->InitSalinityFID);
    
    for(i=0;i<Nc;i++) {
      z = 0;
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
        z-=grid->dz[k]/2;
	phys->s[i][k] = 0;
        phys->s0[i][k] = 0;

        dist_p1 = 100000;
        s_p1 = 0;
        x_p1 = 0;
        y_p1 = 0;
        z_p1 = 0;
	
        if((grid->xv[i]>540888.0 && grid->xv[i]<600443.0 && grid->yv[i]>4187328.0) || (grid->xv[i]>549200.0 && grid->yv[i]<=41872328.0)) {
          for(j=0;j<index;j++) {
            dist = sqrt(pow((phys->xsal[j]-grid->xv[i]),2)+pow((phys->ysal[j]-grid->yv[i]),2)+pow((phys->zsal[j]-z),2));
	    
	    if(dist<dist_p1) {
              s_p1 = phys->sal[j];
              x_p1 = phys->xsal[j];
              y_p1 = phys->ysal[j];
              z_p1 = phys->zsal[j];
              dist_p1 = dist;
            } 
	 
	  }
	  
	  phys->s[i][k] = s_p1;
	  phys->s0[i][k] = s_p1;

	  z-=grid->dz[k]/2;
	}
        else {
          // station 19
          // if(grid->xv[i]<=549200.0) {
            phys->s[i][k] = 32;
            phys->s0[i][k] = 32;
	    // }
        }	
      }
    }

    /*
    stmp = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"InitializePhysicalVariables");
    fread(stmp,sizeof(REAL),grid->Nkmax,prop->InitSalinityFID);
    fclose(prop->InitSalinityFID);

    for(i=0;i<Nc;i++) 
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	phys->s[i][k]=stmp[k];
	phys->s0[i][k]=stmp[k];
      }
    SunFree(stmp,grid->Nkmax,"InitializePhysicalVariables");
    */

  } else {
    for(i=0;i<Nc;i++) {
      z = 0;
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	z-=grid->dz[k]/2;
	phys->s[i][k]=ReturnSalinity(grid->xv[i],grid->yv[i],z);
	phys->s0[i][k]=ReturnSalinity(grid->xv[i],grid->yv[i],z);

	z-=grid->dz[k]/2;
      }
    }
  }

  if(prop->readTemperature) {
    stmp = (REAL *)SunMalloc(grid->Nkmax*sizeof(REAL),"InitializePhysicalVariables");
    fread(stmp,sizeof(REAL),grid->Nkmax,prop->InitTemperatureFID);
    fclose(prop->InitTemperatureFID);    

    for(i=0;i<Nc;i++) 
      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	phys->T[i][k]=stmp[k];

    SunFree(stmp,grid->Nkmax,"InitializePhysicalVariables");
  } else {
    for(i=0;i<Nc;i++) {
      z = 0;
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	z-=grid->dz[k]/2;
	phys->T[i][k]=ReturnTemperature(grid->xv[i],grid->yv[i],z,grid->dv[i]);
	z-=grid->dz[k]/2;
      }
    }
  }
  
  // Initialize the velocity field 
  for(j=0;j<grid->Ne;j++) {
    z = 0;
    phys->de[j]=0.0;
    phys->tau_T[j] = 0.0;
    for(k=0;k<grid->Nke[j];k++) {
      z-=grid->dz[k]/2;
      phys->u[j][k]=ReturnHorizontalVelocity(grid->xe[j],grid->ye[j],grid->n1[j],grid->n2[j],z);
      z-=grid->dz[k]/2;
    }
    //for(k=0;k<grid->Nke[j];k++)
    // phys->wf[j][k]=0;
  }
  // set initial w with continuity
  Continuity(phys->w,grid,phys,prop);

  // Need to compute the velocity vectors at the cell centers based
  // on the initialized velocities at the faces.
  ComputeVelocityVector(phys,phys->uc,phys->vc,grid);
  ComputeVelocityVector(phys,phys->uold,phys->vold,grid);


  // Determine minimum and maximum scalar values.
  phys->smin=phys->s[0][0];
  phys->smax=phys->s[0][0];
  for(i=0;i<grid->Nc;i++)
    for(k=0;k<grid->Nk[i];k++) {
      if(phys->s[i][k]<phys->smin) phys->smin=phys->s[i][k];      
      if(phys->s[i][k]>phys->smax) phys->smax=phys->s[i][k]; 

    }

 
  // Set the density from s and T using the equation of state 
  SetDensity(grid,phys,sedi,prop,sprop);


  // Initialize the eddy-viscosity and scalar diffusivity
  for(i=0;i<grid->Nc;i++) 
    for(k=0;k<grid->Nk[i];k++) {
      phys->nu_tv[i][k]=0;
      phys->kappa_tv[i][k]=0;
    }

  if(prop->turbmodel) 
    for(i=0;i<grid->Nc;i++) 
      for(k=0;k<grid->Nk[i];k++) {
	phys->qT[i][k]=0;
	phys->lT[i][k]=0;
      }

}

/*
 * Function: SetDragcoefficients
 * Usage: SetDragCoefficents(grid,phys,prop);
 * ------------------------------------------
 * Set the drag coefficients based on the log law as well as the applied shear stress.
 *
 */
void SetDragCoefficients(gridT *grid, physT *phys, propT *prop) {
  int i, j, k,jptr, nc1, nc2, kb;
  REAL z0,H, dz1,dz2,zb;

  if(prop->z0T==0) 
    for(j=0;j<grid->Ne;j++) 
      phys->CdT[j]=prop->CdT;
  else
    for(j=0;j<grid->Ne;j++) {
      nc1=grid->grad[2*j];
      nc2=grid->grad[2*j+1];
      if(nc1==-1) nc1=nc2; 
      if(nc2==-1) nc2=nc1; 
      if(grid->Nk[nc2]>grid->Nk[nc1]) nc1=nc2;
      if(grid->Nk[nc1]>grid->Nk[nc2]) nc2=nc1;
     
      phys->CdT[j]=pow(log(0.25*(grid->dzz[nc1][grid->ctop[nc1]]+grid->dzz[nc2][grid->ctop[nc2]])/prop->z0T)/KAPPA_VK,-2);
    }


  if(prop->z0B==0) 
    for(j=0;j<grid->Ne;j++) 
      phys->CdB[j]=prop->CdB;
  else {
    for(j=0;j<grid->Ne;j++) {
      nc1 = grid->grad[2*j];
      nc2 = grid->grad[2*j+1];
      if(nc1==-1) nc1=nc2;
      if(nc2==-1) nc2=nc1;
      kb = grid->Nke[j]-1;

      // find the distance of bottom-most flux center from the bed, zb, for computing drag coefficient
      // if the flux face is only wet on one side, use the flux height on the wet side
      // otherwise use the upwide flux height
      dz1 = grid->dzz[nc1][kb];
      dz2 = grid->dzz[nc2][kb];
      if(dz1==0)
	zb=0.5*dz2;
      else if (dz2==0)
	zb=0.5*dz1;
      else
	zb = 0.5*UpWind(phys->u[j][kb],grid->dzz[nc1][kb],grid->dzz[nc2][kb]);
      
      // determine if the flux face should be considered dry based on the total water depth on the sides
      // use the minimal depth
      dz1 = grid->dv[nc1]+phys->h[nc1];
      dz2 = grid->dv[nc2]+phys->h[nc2];
      if(dz1<dz2)
	H = dz1;
      else
	H = dz2;

      z0 = 1e-6;      

      if(H<0.1)
        phys->CdB[j] = 5;
      else
        phys->CdB[j] = pow(log(zb/z0)/KAPPA_VK,-2);
      
    }
  }



}

/*
 * Function: InitializeVerticalGrid
 * Usage: InitializeVerticalGrid(grid);
 * ------------------------------------
 * Initialize the vertical grid by allocating space for grid->dzz and grid->dzzold
 * This just sets dzz and dzzold to dz since upon initialization dzz and dzzold
 * do not vary in the horizontal.
 *
 */
void InitializeVerticalGrid(gridT **grid)
{
  int i, k, j, Nc=(*grid)->Nc, Ne=(*grid)->Ne;
  int jp[]={1,2,0}, jm[]={2,0,1},nf;
  REAL xt[3],yt[3],ang;
  
  (*grid)->dzz = (REAL **)SunMalloc(Nc*sizeof(REAL *),"InitializeVerticalGrid");
  (*grid)->dzzold = (REAL **)SunMalloc(Nc*sizeof(REAL *),"InitializeVerticalGrid");
 
  (*grid)->dzf = (REAL **)SunMalloc(Ne*sizeof(REAL *),"InitializeVerticalGrid");
  (*grid)->dzfold = (REAL **)SunMalloc(Ne*sizeof(REAL *),"InitializeVerticalGrid");
  
  for(i=0;i<Nc;i++) {
    (*grid)->dzz[i]=(REAL *)SunMalloc(((*grid)->Nk[i])*sizeof(REAL),"InitializeVerticalGrid");
    (*grid)->dzzold[i]=(REAL *)SunMalloc(((*grid)->Nk[i])*sizeof(REAL),"InitializeVerticalGrid");   
    for(k=0;k<(*grid)->Nk[i];k++) {
      (*grid)->dzz[i][k]=(*grid)->dz[k];  
      (*grid)->dzzold[i][k]=(*grid)->dz[k];  
    }
  }

  /* The flux heights are defined down to Nkc, but for any k >= Nke[j], they are always 0. */

  for (j = 0; j < Ne; j++){
    (*grid)->dzf[j] = (REAL *)SunMalloc(((*grid)->Nkc[j])*sizeof(REAL), "InitializeVerticalGrid");
    (*grid)->dzfold[j] = (REAL *)SunMalloc(((*grid)->Nkc[j])*sizeof(REAL), "InitializeVerticalGrid");
    for(k = 0; k < (*grid)->Nke[j]; k++){
      if (k < (*grid)->Nke[j])
	(*grid)->dzf[j][k] = (*grid)->dz[k];
      else
	(*grid)->dzf[j][k] = 0.0;
      (*grid)->dzfold[j][k] = (*grid)->dzf[j][k];
    }
  }
}

/*
 * Function: UpdateDZ
 * Usage: UpdateDZ(grid,phys,0);
 * -----------------------------
 * This function updates the vertical grid spacings based on the free surface and
 * the bottom bathymetry.  That is, if the free surface cuts through cells and leaves
 * any cells dry (or wet), then this function will set the vertical grid spacing 
 * accordingly.
 *
 * If option==1, then it assumes this is the first call and sets dzzold to dzz at
 *   the end of this function.
 * Otherwise it sets dzzold to dzz at the beginning of the function and updates dzz
 *   thereafter.
 *
 */
static void UpdateDZ(gridT *grid, physT *phys, int option)
{
  int i, j, k, ne1, ne2, Nc=grid->Nc, Ne=grid->Ne, flag;
  int verb=0;
  REAL z,dzmin=1e-2, dpth; //1e-2;

  // If this is not an initial call then set dzzold to store the old value of dzz
  // and also set the etopold and ctopold pointers to store the top indices of
  // the grid.
  if(!option) {
    for(j=0;j<Ne;j++){
      grid->etopold[j]=grid->etop[j];
      for(k=0; k<grid->Nke[j]; k++)
	grid->dzfold[j][k] = grid->dzf[j][k];
    }
    for(i=0;i<Nc;i++) {
      grid->ctopold[i]=grid->ctop[i];
      for(k=0;k<grid->Nk[i];k++)
	grid->dzzold[i][k]=grid->dzz[i][k];
    }
  }

  // First set the thickness of the bottom grid layer.  If this is a partial-step
  // grid then the dzz will vary over the horizontal at the bottom layer.  Otherwise,
  // the dzz at the bottom will be equal to dz at the bottom.
  for(i=0;i<Nc;i++) {
    z = 0;
    for(k=0;k<grid->Nk[i];k++)
      z-=grid->dz[k];
    grid->dzz[i][grid->Nk[i]-1]=grid->dz[grid->Nk[i]-1]+grid->dv[i]+z;
  }
  
  // Loop through and set the vertical grid thickness when the free surface cuts through 
  // a particular cell.
  if(grid->Nkmax>1) {
    for(i=0;i<Nc;i++) {
      z = 0;
      flag = 0;
      for(k=0;k<grid->Nk[i];k++) {
	z-=grid->dz[k];
	if(phys->h[i]-z>=dzmin){
	  if(!flag) {
	    if(phys->h[i]==z) {
	      grid->dzz[i][k]=0;
	      grid->ctop[i]=k+1;
	    }else{
  	      if(k==grid->Nk[i]-1) {
  	      	grid->dzz[i][k]=phys->h[i]+grid->dv[i];
  	        grid->ctop[i]=k;
  	        if(grid->dzz[i][k]<=2*dzmin) { 
		  if(grid->dzz[i][k]<dzmin) 
		    phys->h[i]=dzmin-grid->dv[i];
		  grid->dzz[i][k]=0;
		  grid->ctop[i]=k+1;
		}
  	      } else {
  	        grid->dzz[i][k]=phys->h[i]-z;
  	        grid->ctop[i]=k;
  	      }
	    }
	    flag=1;
	  } else {
	    if(k==grid->Nk[i]-1) 
	      grid->dzz[i][k]=grid->dz[k]+grid->dv[i]+z;
	    else 
	      if(z<-grid->dv[i])
		grid->dzz[i][k]=0;
	      else 
		grid->dzz[i][k]=grid->dz[k];
	  } 
	}else
	  grid->dzz[i][k]=0;

      }
      if (!flag){
        grid->ctop[i]=grid->Nk[i];
        phys->h[i]=-grid->dv[i]+dzmin;
      }
    }
  } else 
    for(i=0;i<Nc;i++) 
      grid->dzz[i][0]=grid->dv[i]+phys->h[i];

  // debug updatedz
  for(i=0;i<Nc;i++)
    if(grid->dv[i]+phys->h[i]<0)
      printf("WARNING: negative depth %f, at cell i %d, layers Nk[i] %d\n",grid->dv[i]+phys->h[i],i,grid->Nk[i]);

  // Now set grid->etop and ctop which store the index of the top cell  
  //for(j=0;j<grid->Ne;j++) {
  //   ne1 = grid->grad[2*j];
  //   ne2 = grid->grad[2*j+1];
  //   if(ne1 == -1)
  //     grid->etop[j]=grid->ctop[ne2];
  //    else if(ne2 == -1)
  //      grid->etop[j]=grid->ctop[ne1];
  //    else if(grid->ctop[ne1]<grid->ctop[ne2])
  //     grid->etop[j]=grid->ctop[ne1];
  //   else
  //     grid->etop[j]=grid->ctop[ne2];
  //  }


  // Copy Rusty's code for obtaining dzf
  for (j=0; j<grid->Ne; j++){
    ne1 = grid->grad[2*j];
    ne2 = grid->grad[2*j+1];
    dpth = 0;
    if(ne1 == -1 || ne2 == -1) { //either side is dry

      if (ne1 == -1) ne1=ne2; // set ne1 to be the wet cell
      grid->etop[j] = grid->ctop[ne1];

      for (k=0; k<grid->ctop[ne1] && k<grid->Nke[j]; k++) 
	// set the flux face height above the top wet cell to be zero
	grid->dzf[j][k] = 0.0;

      for (k=grid->ctop[ne1]; k<grid->Nke[j]; k++){
	// set the flux face height in the water column to be the height of the wet cells 
	grid->dzf[j][k] = grid->dzz[ne1][k];
	dpth += grid->dzf[j][k];
      }
           
      if (grid->etop[j] > grid->Nke[j])
	grid->etop[j] = grid->Nke[j];
    }else{
      if (!option && grid->etop[j] < grid->Nke[j] && phys->u[j][grid->etop[j]] != 0.0){
	if (verb)
	  printf("UpdateDz: edge %d is wet and velocity is nonzero\n", j);
	if (phys->u[j][grid->etop[j]] > 0)
	  ne1=ne2; // ne2 is the upwind cell. Otherwise ne1 already refers to the upwind cell	
      }else{
	if (verb)
	  printf("UpdateDz: edge %d is dry or zero velocity\n", j);
	if(phys->h[ne2] > phys->h[ne1])
	  ne1 = ne2;
      }
      // set the top cell index to be the one in the upwind or in the higher water level (if the
      // edge is dry), but can not be lower than the edge bottom.
      grid->etop[j] = Min(grid->ctop[ne1],grid->Nke[j]);

      if (verb)
	printf("UpdateDz: taking top if edge %d from cell %d, set etop=%d\n", j, ne1, grid->etop[j]);

      if (grid->ctop[ne1] == grid->Nk[ne1]){
	grid->etop[j] = grid->Nke[j];
	if (verb) printf("UpdateDz: but that cell is dry. Drying this edge\n");
      }
      
      if (grid->etop[j] == grid->Nke[j]){ //totally dry case
	for(k=0; k<grid->Nke[j]; k++)
	  grid->dzf[j][k] = 0.0;
      }else{
	for (k=0; k< grid->etop[j]; k++)
	  grid->dzf[j][k] = 0.0;
	for (k=grid->etop[j]; k<grid->Nke[j]; k++){
	  if (k == grid->etop[j])
	    grid->dzf[j][k] = grid->dzz[ne1][k];
	  else
	    grid->dzf[j][k] = grid->dz[k];
	  dpth += grid->dzf[j][k];
	}
      }      
    }
    phys->de[j] = dpth;
    

  }
  // If this is an initial call set the old values to the new values.
  if(option) {
    for(j=0;j<Ne;j++){ 
      grid->etopold[j]=grid->etop[j];
      for (k=0; k<grid->Nke[j]; k++)
	grid->dzfold[j][k] = grid->dzf[j][k];
    }  
    for(i=0;i<Nc;i++) {
      grid->ctopold[i]=grid->ctop[i];
      for(k=0;k<grid->Nk[i];k++)
	grid->dzzold[i][k]=grid->dzz[i][k];
    }
  }

}

/*
 * Function: Solve
 * Usage: Solve(grid,phys,prop,myproc,numprocs,comm);
 * --------------------------------------------------
 * This is the main solving routine and is called from suntans.c.
 * solvefunction
 *
 */
void Solve(gridT *grid, physT *phys, propT *prop, spropT *sprop, sediT *sedi, 
           waveT *wave, wpropT *wprop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, j, k, n, blowup=0, s;
  extern int TotSpace;
  int procp=0, ne,nf,edgesp[3],normp[3],cp=692,jp=148,kp=22;
  double f[3],vol=0.0,htemp,utheta;
  int numbercells=0;

  for(i=0;i<grid->Nc;i++)
    numbercells+=grid->Nk[i];
  printf("proc %d: number of cells %d\n",myproc,numbercells);
 

  // Compute the initial quantities for comparison to determine conservative properties
  prop->n=0; 
  ComputeConservatives(grid,phys,prop,myproc,numprocs,comm);

  // Print out memory usage per processor if this is the first time step
  if(VERBOSE>1) printf("Processor %d,  Total memory: %d Mb\n",myproc,(int)(TotSpace/(1024*1e3)));

  prop->theta0=prop->theta;

  // Output initial conditions

  OutputData(grid, phys, prop, sedi, sprop, wave, wprop, myproc, numprocs, blowup, comm);
 
  for(n=prop->nstart+1;n<=prop->nsteps+prop->nstart;n++) {
  

    prop->n = n;
    prop->rtime = (n-1)*prop->dt;

    MPI_Barrier(comm);
    if(prop->nsteps>0) {

      // Set boundary values

      BoundaryVelocities(grid,phys,prop,myproc);
      
      //BoundaryScalars(grid,phys,prop);
      //WindStress(grid,phys,prop);
      
      SetDragCoefficients(grid,phys,prop);
      
      // Ramp down theta from 1 to the value specified in suntans.dat over
      // the time thetaramptime specified in suntans.dat to damp out transient
      // oscillations
      if(prop->thetaramptime!=0)
	prop->theta=(1-exp(-prop->rtime/prop->thetaramptime))*prop->theta0+
	  exp(-prop->rtime/prop->thetaramptime);
      
      // Store the old velocity and scalar fields
      StoreVariables(grid,phys);
      

      HorizontalSource(grid,phys,prop,wave,wprop,myproc,numprocs,comm);
      
      // Use the explicit part created in HorizontalSource and solve for the free-surface
      // and hence compute the predicted or hydrostatic horizontal velocity field.  Then
      // send and receive the free surface interprocessor boundary data to the neighboring processors.
      // The predicted horizontal velocity is now in phys->u
 

      UPredictor(grid,phys, prop,myproc,numprocs,comm);
      
      
      ISendRecvCellData2D(phys->h,grid,myproc,comm);
      
      // Adjust the velocity field in the new cells if the newcells variable is set to 1 in
      // suntans.dat.  Once this is done, send the interprocessor u-velocities to the neighboring
      // processors.
      ISendRecvEdgeData3D(phys->u,grid,myproc,comm);
      
      // Compute vertical momentum and the nonhydrostatic pressure
      if(prop->nonhydrostatic) {

	// Predicted vertical velocity field is in phys->w
	WPredictor(grid,phys,prop,myproc,numprocs,comm);

	// Source term for the pressure-Poisson equation is in phys->stmp
	ComputeQSource(phys->stmp,grid,phys,prop,myproc,numprocs);

	// Solve for the nonhydrostatic pressure.  
	// phys->stmp2 contains the initial guess
	// phys->stmp contains the source term
	// phys->stmp3 is used for temporary storage
	CGSolveQ(phys->qc,phys->stmp,phys->stmp3,grid,phys,prop,myproc,numprocs,comm);
	
	// Correct the nonhydrostatic velocity field with the nonhydrostatic pressure
	// correction field phys->stmp2.  This will correct phys->u so that it is now
	// the volume-conserving horizontal velocity field.  phys->w is not corrected since
	// it is obtained via continuity.  Also, update the total nonhydrostatic pressure
	// with the pressure correction. 
	Corrector(phys->qc,grid,phys,prop,myproc,numprocs,comm);

	// Send/recv the horizontal velocity data after it has been corrected.
	ISendRecvEdgeData3D(phys->u,grid,myproc,comm);
	// Send q to the boundary cells now that it has been updated
	ISendRecvCellData3D(phys->q,grid,myproc,comm);
      }	
      // Compute the vertical velocity based on continuity and then send/recv to
      // neighboring processors.
      Continuity(phys->w,grid,phys,prop);

      // Send/recv the vertical velocity data 
      ISendRecvWData(phys->w,grid,myproc,comm);
      

      // Compute the eddy viscosity
      EddyViscosity(grid,phys,sedi,prop,comm,myproc);

      // Update the salinity only if beta is nonzero in suntans.dat


      if(prop->beta) {

	UpdateScalars(grid,phys,prop,phys->s,phys->boundary_s,phys->Cn_R,prop->kappa_s,prop->kappa_sH,phys->kappa_tv,prop->thetaS,
		      NULL,NULL,NULL,NULL,0,0,comm,myproc);

	ISendRecvCellData3D(phys->s,grid,myproc,comm);
      }

      // Update the temperature only if gamma is nonzero in suntans.dat
      if(prop->gamma) {
	UpdateScalars(grid,phys,prop,phys->T,phys->boundary_T,phys->Cn_T,prop->kappa_T,prop->kappa_TH,phys->kappa_tv,prop->thetaS,
		      NULL,NULL,NULL,NULL,0,0,comm,myproc);
	ISendRecvCellData3D(phys->T,grid,myproc,comm);
      }


      
      //     if (prop->n % wprop->nwind == 1)

      if(prop->beta || prop->gamma)
      	SetDensity(grid,phys,sedi,prop,sprop);

    }
    // utmp2 contains the velocity field at time step n, u contains
    // it at time step n+1.  This is so that at the next time step
    // phys->uold contains velocity at time step n-1 and phys->uc contains
    // that at time step n.                                             
    ComputeVelocityVector(phys,phys->uc,phys->vc,grid);
    ISendRecvCellData3D(phys->uc,grid,myproc,comm);
    ISendRecvCellData3D(phys->vc,grid,myproc,comm);

    if (prop->wave)
      if ((prop->n-1) % wprop->wnstep == 0){
    	if (wprop->wind_forcing){
    	  WindField(prop, grid, phys, wave, wprop, myproc, numprocs);
    	}
	UpdateWave(grid, phys, wave, prop, wprop, sedi, sprop, comm, myproc, numprocs); //tsungwei20140621	
      }
    
//............................/ycshao2015 only for spin out    

    if (prop->sedi)
      CalculateBedSediment(grid, phys, prop, sedi, sprop, wave);
    
    if (prop->sedi){
      for(s = 0; s < sprop->Nsize; s++){
    	UpdateSedi(grid,phys,prop,sedi, sprop, wave, sedi->sd[s],s,phys->boundary_T,phys->Cn_R,prop->kappa_s,prop->kappa_sH,phys->kappa_tv,prop->thetaS,
		   NULL,NULL,NULL,NULL,0,0,comm,myproc);
	 
    	ISendRecvCellData3D(sedi->sd[s],grid,myproc,comm);
      }
      CalculateTotalSediment(grid, sedi, sprop);
    }

//................................./ycshao2015
    // Output progress
    Progress(prop,myproc);       
    // Check whether or not run is blowing up
    blowup=Check(grid,phys,prop,sedi,sprop,wave,wprop,myproc,numprocs,comm);

    // Output data based on ntout specified in suntans.dat
    OutputData(grid,phys,prop,sedi,sprop, wave,wprop,myproc,numprocs,blowup,comm);
    
    InterpData(sedi,grid,phys,prop,comm,numprocs,myproc);

    if(blowup)
      break;

  }
}

/*
 * Function: StoreVariables
 * Usage: StoreVariables(grid,phys);
 * ---------------------------------
 * Store the old values of s, u, and w into stmp3, utmp2, and wtmp2,
 * respectively.
 *
 */
static void StoreVariables(gridT *grid, physT *phys) {
  int i, j, k, iptr, jptr;

  for(i=0;i<grid->Nc;i++) 
    for(k=0;k<grid->Nk[i];k++) {
      phys->stmp3[i][k]=phys->s[i][k];
      phys->wtmp2[i][k]=phys->w[i][k];
    }

  for(j=0;j<grid->Ne;j++) {
    phys->D[j]=0;
    for(k=0;k<grid->Nke[j];k++)
      phys->utmp2[j][k]=phys->u[j][k];
  }
}

/*
 * Function: HorizontalSource
 * Usage: HorizontalSource(grid,phys,prop,myproc,numprocs);
 * --------------------------------------------------------
 * Compute the horizontal source term that is used to obtain the free surface.
 *
 * This function adds the following to the horizontal source term:
 *
 * 1) Old nonhydrostatic pressure gradient with theta method
 * 2) Coriolis terms with AB2
 * 3) Baroclinic term with AB2
 * 4) Horizontal and vertical advection of horizontal momentum with AB2
 * 5) Horizontal laminar+turbulent diffusion of horizontal momentum
 *
 * Cn_U contains the Adams-Bashforth terms at time step n-1.
 * If wetting and drying is employed, no advection is computed in 
 * the upper cell.
 *
 */
static void HorizontalSource(gridT *grid, physT *phys, propT *prop, waveT *wave,
			     wpropT *wprop, int myproc, int numprocs, MPI_Comm comm) {
  int i, iptr, nf, j, jptr, k, nc, nc1, nc2, ne, k0, kmin, kmax;
  REAL *a, *b, *c, fab, sum;
  // Gang elm  
  int lagra_cell, lagra_layer, ncf,ncn,ncd,bc3;
  int inter_proc_cell, close_to_wall, neigh,n, neighproc,neighcell;
  REAL  xs[2],zs,us[2],al[3],vtx[3],vty[3],tmp,vtmp, h0;
 
  int edgesp[3],normp[3];
  int ncp1=76,ncp2=29,jp=148,procp=0,kp=2;
  REAL us_neighlayer[2],alpha;
  
  a = phys->a;
  b = phys->b;
  c = phys->c;

  // fab is 1 for a forward Euler calculation on the first time step,
  // for which Cn_U is 0.  Otherwise, fab=3/2 and Cn_U contains the
  // Adams-Bashforth terms at time step n-1
  if(prop->n==1) {
    fab=1;
    for(j=0;j<grid->Ne;j++)
      for(k=0;k<grid->Nke[j];k++)
	phys->Cn_U[j][k]=0;
  } else
    fab=1.5;

  // Set utmp and ut to zero since utmp will store the source term of the
  // horizontal momentum equation
  for(j=0;j<grid->Ne;j++) {
    for(k=0;k<grid->Nke[j];k++) {
      phys->utmp[j][k]=0;
      phys->ut[j][k]=0;
    }
  }

  // Sponge layer at x=0 that decays exponentially with distance sponge_distance
  // over a timescale given by sponge_decay.  Both defined in suntans.dat
  // First use Cn_U from step n-1 then set it to 0.
  if(prop->sponge_distance==0) {
    if(prop->nonlinear==3){	
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
        j = grid->edgep[jptr];
      
        nc1 = grid->grad[2*j];
        nc2 = grid->grad[2*j+1];
      
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  phys->utmp[j][k]=(1-fab)*phys->Cn_U[j][k]-prop->dt/grid->dg[j]*(phys->q[nc1][k]-phys->q[nc2][k]);
	  phys->Cn_U[j][k]=0;
        }
      }
    }else{
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
        j = grid->edgep[jptr];
      
        nc1 = grid->grad[2*j];
        nc2 = grid->grad[2*j+1];
      
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  phys->utmp[j][k]=(1-fab)*phys->Cn_U[j][k]+phys->u[j][k]-prop->dt/grid->dg[j]*(phys->q[nc1][k]-phys->q[nc2][k]);
	  phys->Cn_U[j][k]=0;
        }
      }
    }


    // Add on explicit term to boundary edges
    for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
      j = grid->edgep[jptr];

      for(k=grid->etop[j];k<grid->Nke[j];k++) {
	phys->utmp[j][k]=(1-fab)*phys->Cn_U[j][k]+phys->u[j][k];

	phys->Cn_U[j][k]=0;
      }
    }
  } else {
    for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
      j = grid->edgep[jptr];

      nc1 = grid->grad[2*j];
      nc2 = grid->grad[2*j+1];

      for(k=grid->etop[j];k<grid->Nke[j];k++) {
	phys->utmp[j][k]=-prop->dt/grid->dg[j]*(phys->q[nc1][k]-phys->q[nc2][k])
	  +(1.0-prop->dt*exp(-0.5*(grid->xv[nc1]+grid->xv[nc2])/prop->sponge_distance)/
	    prop->sponge_decay)*phys->u[j][k];
      
	phys->Cn_U[j][k]=0;
      }
    }
    for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
      j = grid->edgep[jptr];

      nc1 = grid->grad[2*j];

      for(k=grid->etop[j];k<grid->Nke[j];k++) {
	phys->utmp[j][k]=(1.0-prop->dt*exp(-grid->xv[nc1]/prop->sponge_distance)/
	    prop->sponge_decay)*phys->u[j][k];

	phys->Cn_U[j][k]=0;

      }
    }
  }


  // Coriolis terms
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
    for(k=grid->etop[j];k<grid->Nke[j];k++)
      phys->Cn_U[j][k]+=prop->dt*prop->Coriolis_f*(InterpToFace(j,k,phys->vc,phys->u,grid)*grid->n1[j]-
						   InterpToFace(j,k,phys->uc,phys->u,grid)*grid->n2[j]);
  }
  //Radiation Stress
  if (prop->wave){
    if (wprop->rad_stress)
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	j = grid->edgep[jptr];
	//for(j = 0; j < grid->Ne; j++){
	for(k=grid->etop[j];k<grid->Nkc[j];k++){
	  //if (abs(wave->divSe[j][k]) > 0.1)
	  //	printf("-------------------------------j=%d; k=%d; myproc = %d; divSe = %f\n", j, k, myproc, wave->divSe[j][k]);
	  phys->Cn_U[j][k]+=wave->divSe[j][k];
       	
	}
      }
  }

  if (wprop->wind_shear)
    for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
      j = grid->edgep[jptr];
      if(grid->etop[j]<grid->Nke[j]){
	nc1 = grid->grad[2*j];
	nc2 = grid->grad[2*j+1];
      
	if(nc1==-1)
	  nc1=nc2;
	if(nc2==-1)
	  nc2=nc1;
	h0 = 0.5*(grid->dzz[nc1][grid->etop[j]]+grid->dzz[nc2][grid->etop[j]]);
      // Add the shear stress from the top cell
	if (h0 < 0.01)
	  h0 = 0.01;

	phys->Cn_U[j][grid->etop[j]] += phys->tau_T[j]/h0*prop->dt;
      }
    }

  // ************* Baroclinic term ***************//
  //for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {

  //j = grid->edgep[jptr];
	
  // nc1 = grid->grad[2*j];
  //nc2 = grid->grad[2*j+1];
    
    // Add the explicit part of the free-surface to create U**.

  //for(k=grid->etop[j];k<grid->Nke[j];k++){ 
  //  if(phys->Cn_U[j][k]!=phys->Cn_U[j][k]) {
  //printf("......................Proc:%d Error in function at j%d k=%d utmp=%f\n",myproc,j,k,phys->Cn_U[j][k]);
	//	printf("etop=%d Nke=%d h1=%f h2=%f dg=%f\n",grid->etop[j],grid->Nke[j],phys->h[nc1],phys->h[nc2],grid->dg[j]);
	    
  //   }

  // }
      
  //}  



  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
    if(grid->etop[j]<grid->Nke[j]-1)
      for(k=grid->etop[j];k<grid->Nke[j];k++) {
	k0=grid->etop[j];

	for(k0=grid->etop[j];k0<k;k0++)
	  phys->Cn_U[j][k]-=0.5*GRAV*prop->dt*(phys->rho[nc1][k0]-phys->rho[nc2][k0])*
	    (grid->dzz[nc1][k0]+grid->dzz[nc2][k0])/grid->dg[j];
	phys->Cn_U[j][k]-=0.25*GRAV*prop->dt*(phys->rho[nc1][k]-phys->rho[nc2][k])*
	  (grid->dzz[nc1][k]+grid->dzz[nc2][k])/grid->dg[j];



	//if(phys->Cn_U[j][k]!=phys->Cn_U[j][k]) {
	//  printf("......................Proc:%d Error at j%d k=%d nc1=%d nc2=%d utmp=%f\n",myproc,j,k,nc1,nc2,phys->Cn_U[j][k]);
	//  printf("etop=%d Nke=%d rho1=%f rho2=%f dg=%f\n",grid->etop[j],grid->Nke[j],phys->rho[nc1][k],phys->rho[nc2][k],grid->dg[j]);
	//}

      }
  }




  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      k0=grid->etop[j];

      for(k0=grid->etop[j];k0<k;k0++)
	phys->Cn_U[j][k]-=GRAV*prop->dt*(phys->rho[nc1][k0]-phys->boundary_rho[jptr-grid->edgedist[2]][k0])*
	  grid->dzz[nc1][k0]/grid->dg[j];
      phys->Cn_U[j][k]-=0.5*GRAV*prop->dt*(phys->rho[nc1][k]-phys->boundary_rho[jptr-grid->edgedist[2]][k])*
	grid->dzz[nc1][k]/grid->dg[j];
    }
  }

  // Set stmp and stmp2 to zero since these are used as temporary variables 
  //for advection and diffusion.
  for(i=0;i<grid->Nc;i++)
    for(k=0;k<grid->Nk[i];k++)
      phys->stmp[i][k]=phys->stmp2[i][k]=0;

  // ****** Compute Eulerian advection of momentum (nonlinear!=0) ******//
  if(prop->nonlinear) {
    // Compute the u-component fluxes at the faces
    if(prop->nonlinear==3){  // nonlinear=3 ELM
      
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	j = grid->edgep[jptr];

	nc1 = grid->grad[2*j];
	nc2 = grid->grad[2*j+1];

	for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  if(phys->u[j][k]>=0)
	    nc=nc2;
	  else
	    nc=nc1;
	  if(grid->dzz[nc][k]>0){
  	    xs[0]=grid->xe[j];
  	    xs[1]=grid->ye[j];
	    zs=grid->dzz[nc][k]/2;
	 	 
	    LagraTracing(grid, phys, prop, nc, k, &(lagra_cell), &(lagra_layer), xs,&zs,myproc);
	    
	    // added for conservation
	    inter_proc_cell=0;
	    for(neigh=0;neigh<grid->Nneighs;neigh++) {
	      if(!inter_proc_cell) {
		neighproc = grid->myneighs[neigh];
	       
		for(n=0;n<grid->num_cells_recv[neigh];n++)
		  if(lagra_cell==grid->cell_recv[neigh][n]) {
		    inter_proc_cell = 1;
		    break;
		  }
	      }
	      else
		break;
	    }

	    //InterpVelo(grid,phys, lagra_cell, lagra_layer,xs,us,al,vtx,vty,0);
	    //StableInterpVelo(grid,phys, lagra_cell, lagra_layer,xs,us);
	    
	    // interp horizontal velocity with 2 layers
	    
	    if(zs>0.5*grid->dzz[lagra_cell][lagra_layer] && lagra_layer+1<grid->Nk[lagra_cell] && !inter_proc_cell) {
	      StableInterpVelo(grid, phys, lagra_cell, lagra_layer, xs, us);
	      StableInterpVelo(grid, phys, lagra_cell, lagra_layer+1, xs, us_neighlayer);
	      
	      alpha = (zs-0.5*grid->dzz[lagra_cell][lagra_layer])/(0.5*grid->dzz[lagra_cell][lagra_layer]+0.5*grid->dzz[lagra_cell][lagra_layer+1]);
	      us[0]=(1.0-alpha)*us[0]+alpha*us_neighlayer[0];
	      us[1]=(1.0-alpha)*us[1]+alpha*us_neighlayer[1];
	    } 
	    else if(zs<0.5*grid->dzz[lagra_cell][lagra_layer] && lagra_layer-1>grid->ctop[lagra_cell] && !inter_proc_cell) {
	      StableInterpVelo(grid, phys, lagra_cell, lagra_layer, xs, us);
	      StableInterpVelo(grid, phys, lagra_cell, lagra_layer-1, xs, us_neighlayer);
	      
	      alpha = (0.5*grid->dzz[lagra_cell][lagra_layer]-zs)/(0.5*grid->dzz[lagra_cell][lagra_layer]+0.5*grid->dzz[lagra_cell][lagra_layer-1]);
	      us[0]=(1.0-alpha)*us[0]+alpha*us_neighlayer[0];
	      us[1]=(1.0-alpha)*us[1]+alpha*us_neighlayer[1];
	    }   
	    else 
	      if(inter_proc_cell)
		InterpVelo(grid,phys, lagra_cell, lagra_layer, xs, us, al, vtx, vty,0);
	      else
		StableInterpVelo(grid, phys, lagra_cell, lagra_layer, xs, us);
	    

	    phys->utmp[j][k] += us[0]*grid->n1[j]+us[1]*grid->n2[j];

	  }
	  else
	    phys->utmp[j][k]+=phys->u[j][k];
	}
      }
    }   // end nonlinear=3
    else {                     //nonlinear=1,2
      if(prop->nonlinear==1)  {// nonlinear=1 Upwind
	for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	  j = grid->edgep[jptr];
	  
	  nc1 = grid->grad[2*j];
	  nc2 = grid->grad[2*j+1];
	  
	  if(grid->ctop[nc1]>grid->ctop[nc2])
	    kmin = grid->ctop[nc1];
	  else
	    kmin = grid->ctop[nc2];
	  
	  for(k=0;k<kmin;k++)
	    phys->ut[j][k]=0;
	  
	  for(k=kmin;k<grid->Nke[j];k++) {
	    if(phys->u[j][k]>0)
	      nc=nc2;
	    else
	      nc=nc1;
	    phys->ut[j][k]=phys->uc[nc][k]*grid->dzz[nc][k];
	  }
	}
      } 
      else {            // nonlinear=2 Central
	for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	  j = grid->edgep[jptr];
	  
	  nc1 = grid->grad[2*j];
	  nc2 = grid->grad[2*j+1];
	  
	  if(grid->ctop[nc1]>grid->ctop[nc2])
	    kmin = grid->ctop[nc1];
	  else
	    kmin = grid->ctop[nc2];
	  
	  for(k=0;k<kmin;k++)
	    phys->ut[j][k]=0;
	  
	  for(k=kmin;k<grid->Nke[j];k++) 
	    phys->ut[j][k]=0.5*InterpToFace(j,k,phys->uc,phys->u,grid)*(grid->dzz[nc1][k]+grid->dzz[nc2][k]);
	}
      }

      // Now compute the cell-centered source terms and put them into stmp
      for(i=0;i<grid->Nc;i++) {
	
	for(nf=0;nf<NFACES;nf++) {
	  
	  ne = grid->face[i*NFACES+nf];
	  
	  for(k=grid->ctop[i]+1;k<grid->Nk[i];k++)
	    phys->stmp[i][k]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	      (grid->Ac[i]*grid->dzz[i][k]);
	  
	  // Top cell is filled with momentum from neighboring cells
	  for(k=grid->etop[ne];k<=grid->ctop[i];k++) 
	    phys->stmp[i][grid->ctop[i]]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	      (grid->Ac[i]*grid->dzz[i][grid->ctop[i]]);
	}
      }     
      
      // V-fluxes at boundary cells (was here)
      
      // Compute the v-component fluxes at the faces
      if(prop->nonlinear==1)  // Upwind
        for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	  j = grid->edgep[jptr];
	  
	  nc1 = grid->grad[2*j];
	  nc2 = grid->grad[2*j+1];
	  
	  if(grid->ctop[nc1]>grid->ctop[nc2])
	    kmin = grid->ctop[nc1];
	  else
	    kmin = grid->ctop[nc2];
	  
	  for(k=0;k<kmin;k++)
	    phys->ut[j][k]=0;
	  for(k=kmin;k<grid->Nke[j];k++) {
	    if(phys->u[j][k]>0)
	      nc=nc2;
	    else
	      nc=nc1;
	    phys->ut[j][k]=phys->vc[nc][k]*grid->dzz[nc][k];
	  }
        }
      else // Central
        for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	  j = grid->edgep[jptr];
	  
	  nc1 = grid->grad[2*j];
	  nc2 = grid->grad[2*j+1];
	  
	  if(grid->ctop[nc1]>grid->ctop[nc2])
	    kmin = grid->ctop[nc1];
	  else
	    kmin = grid->ctop[nc2];
	  
	  for(k=0;k<kmin;k++)
	    phys->ut[j][k]=0;
	  
	  for(k=kmin;k<grid->Nke[j];k++) 
	    phys->ut[j][k]=0.5*InterpToFace(j,k,phys->vc,phys->u,grid)*(grid->dzz[nc1][k]+grid->dzz[nc2][k]);
        }
      
      // Now compute the cell-centered source terms and put them into stmp.
      for(i=0;i<grid->Nc;i++) {
        
        for(k=0;k<grid->Nk[i];k++) 
	  phys->stmp2[i][k]=0;
        
        for(nf=0;nf<NFACES;nf++) {
	  
	  ne = grid->face[i*NFACES+nf];
	  
	  for(k=grid->ctop[i]+1;k<grid->Nk[i];k++)
	    phys->stmp2[i][k]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	      (grid->Ac[i]*grid->dzz[i][k]);
	  
	  // Top cell is filled with momentum from neighboring cells
	  for(k=grid->etop[ne];k<=grid->ctop[i];k++) 
	    phys->stmp2[i][grid->ctop[i]]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	      (grid->Ac[i]*grid->dzz[i][grid->ctop[i]]);
        }
      }
    
      // Now do vertical advection
      for(i=0;i<grid->Nc;i++) {
        
        if(prop->nonlinear==1)  // Upwind
	  for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
	    a[k] = 0.5*((phys->w[i][k]+fabs(phys->w[i][k]))*phys->uc[i][k]+
	  	      (phys->w[i][k]-fabs(phys->w[i][k]))*phys->uc[i][k-1]);
	    b[k] = 0.5*((phys->w[i][k]+fabs(phys->w[i][k]))*phys->vc[i][k]+
	  	      (phys->w[i][k]-fabs(phys->w[i][k]))*phys->vc[i][k-1]);
	  }
        else  // Central
	  for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
	    a[k] = phys->w[i][k]*((grid->dzz[i][k-1]/(grid->dzz[i][k]+grid->dzz[i][k-1])*phys->uc[i][k]+
	  			 grid->dzz[i][k]/(grid->dzz[i][k]+grid->dzz[i][k-1])*phys->uc[i][k-1]));
	    b[k] = phys->w[i][k]*((grid->dzz[i][k-1]/(grid->dzz[i][k]+grid->dzz[i][k-1])*phys->vc[i][k]+
	  			 grid->dzz[i][k]/(grid->dzz[i][k]+grid->dzz[i][k-1])*phys->vc[i][k-1]));
	  }
        
        for(k=grid->ctop[i]+1;k<grid->Nk[i]-1;k++) {
	  phys->stmp[i][k]+=(a[k]-a[k+1])/grid->dzz[i][k];
	  phys->stmp2[i][k]+=(b[k]-b[k+1])/grid->dzz[i][k];
        }
        
        if(grid->ctop[i]!=grid->Nk[i]-1) {
	  // Top - advection only comes in through bottom of cell.
	  phys->stmp[i][grid->ctop[i]]-=a[grid->ctop[i]+1]/grid->dzz[i][grid->ctop[i]];
	  phys->stmp2[i][grid->ctop[i]]-=b[grid->ctop[i]+1]/grid->dzz[i][grid->ctop[i]];
	  // Bottom - advection only comes in through top of cell.
	  phys->stmp[i][grid->Nk[i]-1]+=a[grid->Nk[i]-1]/grid->dzz[i][grid->Nk[i]-1];
	  phys->stmp2[i][grid->Nk[i]-1]+=b[grid->Nk[i]-1]/grid->dzz[i][grid->Nk[i]-1];
        }
      }
    } //End nonlinear=1 or 2
  } //End nonlinear term
  
  // ****** Now add on horizontal diffusion to stmp and stmp2 ******//
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];
    
    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
    bc3=0;
    for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
      i = grid->cellp[iptr];
      if(nc1==i ||nc2==i)
        bc3=1;
    }
    if (!bc3){
      if(grid->ctop[nc1]>grid->ctop[nc2])
        kmin = grid->ctop[nc1];
      else
        kmin = grid->ctop[nc2];
      
      for(k=kmin;k<grid->Nke[j];k++) {
        a[k]=prop->nu_H*(phys->uc[nc2][k]-phys->uc[nc1][k])*grid->df[j]/grid->dg[j];
        b[k]=prop->nu_H*(phys->vc[nc2][k]-phys->vc[nc1][k])*grid->df[j]/grid->dg[j];
        phys->stmp[nc1][k]-=a[k]/grid->Ac[nc1];
        phys->stmp[nc2][k]+=a[k]/grid->Ac[nc2];
        phys->stmp2[nc1][k]-=b[k]/grid->Ac[nc1];
        phys->stmp2[nc2][k]+=b[k]/grid->Ac[nc2];
      }
      
      for(k=grid->Nke[j];k<grid->Nk[nc1];k++) {
        phys->stmp[nc1][k]+=prop->CdW*fabs(phys->uc[nc1][k])*phys->uc[nc1][k]*grid->df[j]/grid->Ac[nc1];
        phys->stmp2[nc1][k]+=prop->CdW*fabs(phys->vc[nc1][k])*phys->vc[nc1][k]*grid->df[j]/grid->Ac[nc1];
      }
      for(k=grid->Nke[j];k<grid->Nk[nc2];k++) {
        phys->stmp[nc2][k]+=prop->CdW*fabs(phys->uc[nc2][k])*phys->uc[nc2][k]*grid->df[j]/grid->Ac[nc2];
        phys->stmp2[nc2][k]+=prop->CdW*fabs(phys->vc[nc2][k])*phys->vc[nc2][k]*grid->df[j]/grid->Ac[nc2];
      }
    }
    
  }

  // Check to make sure integrated fluxes are 0 for conservation
  // This will not be conservative if CdW or nu_H are nonzero!
  if(WARNING && prop->CdW==0 && prop->nu_H==0) {
    sum=0;
    for(i=0;i<grid->Nc;i++) {
      for(k=grid->ctop[i];k<grid->Nk[i];k++)
	sum+=grid->Ac[i]*phys->stmp[i][k]*grid->dzz[i][k];
    }
    if(fabs(sum)>CONSERVED) printf("Warning, not U-momentum conservative!\n");

    sum=0;
    for(i=0;i<grid->Nc;i++) {
      for(k=grid->ctop[i];k<grid->Nk[i];k++)
	sum+=grid->Ac[i]*phys->stmp2[i][k]*grid->dzz[i][k];
    }
    if(fabs(sum)>CONSERVED) printf("Warning, not V-momentum conservative!\n");
  }
  
  // Send/recv stmp and stmp2 to account for advective fluxes in ghost cells at
  // interproc boundaries.



  ISendRecvCellData3D(phys->stmp,grid,myproc,comm);
  ISendRecvCellData3D(phys->stmp2,grid,myproc,comm);

     
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr]; 
    
    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
    if(nc1==-1) nc1=nc2;
    if(nc2==-1) nc2=nc1;
    
    if(grid->ctop[nc1]>grid->ctop[nc2])
      k0=grid->ctop[nc1];
    else
      k0=grid->ctop[nc2];
    
    for(k=k0;k<grid->Nk[nc1];k++){ 

      phys->Cn_U[j][k]-=grid->def[nc1*NFACES+grid->gradf[2*j]]/grid->dg[j]
	*prop->dt*(phys->stmp[nc1][k]*grid->n1[j]+phys->stmp2[nc1][k]*grid->n2[j]);
    }
    
    for(k=k0;k<grid->Nk[nc2];k++){ 
      phys->Cn_U[j][k]-=grid->def[nc2*NFACES+grid->gradf[2*j+1]]/grid->dg[j]
	*prop->dt*(phys->stmp[nc2][k]*grid->n1[j]+phys->stmp2[nc2][k]*grid->n2[j]);
    }

  }
 
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr]; 
    
    for(k=grid->etop[j];k<grid->Nke[j];k++){
      phys->utmp[j][k]+=fab*phys->Cn_U[j][k];	

    }
    	  
  }


  // Now add on to the open boundaries
  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr]; 
    
    nc1 = grid->grad[2*j];
    k0=grid->ctop[nc1];

    if(phys->boundary_flag[jptr-grid->edgedist[2]]==open)
      for(k=k0;k<grid->Nk[nc1];k++) 
	phys->Cn_U[j][k]-=0.5*prop->dt*(phys->stmp[nc1][k]*grid->n1[j]+phys->stmp2[nc1][k]*grid->n2[j]);
  }
  
  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr]; 
    
    if(phys->boundary_flag[jptr-grid->edgedist[2]]==open)
      for(k=grid->etop[j];k<grid->Nke[j];k++)
	phys->utmp[j][k]+=fab*phys->Cn_U[j][k];
  }

}


/*
 * Function: NewCells
 * Usage: NewCells(grid,phys,prop);
 * --------------------------------
 * Adjust the velocity in the new cells that were previously dry.
 * This function is required for an Eulerian advection scheme because
 * it is difficult to compute the finite volume form of advection in the
 * upper cells when wetting and drying is employed.  In this function
 * the velocity in the new cells is set such that the quantity u*dz is
 * conserved from one time step to the next.  This works well without
 * wetting and drying.  When wetting and drying is employed, it is best
 * to extrapolate from the lower cells to obtain the velocity in the new
 * cells.
 *
 */
static void NewCells(gridT *grid, physT *phys, propT *prop) {
 
  int j, jptr, k, nc1, nc2,i,iptr;
  REAL dz;
 
  //If the free surface rises up to the next layer. set uc,vc,w and q to be the same as the layer below
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    if(grid->ctop[i]<grid->ctopold[i]){ 
      for(k=grid->ctop[i];k<grid->ctopold[i];k++){
	phys->uc[i][k]=phys->uc[i][grid->ctopold[i]]; 
	phys->vc[i][k]=phys->vc[i][grid->ctopold[i]]; 
	phys->w[i][k]=phys->w[i][grid->ctopold[i]];
	phys->Cn_W[i][k]=0;
      }
      if (prop->nonhydrostatic){
        for(k=grid->ctop[i];k<grid->ctopold[i];k++)
	    phys->q[i][k]=0;	 
        
      }  
    }
  }

}

/*
 * Function: WPredictor
 * Usage: WPredictor(grid,phys,prop,myproc,numprocs,comm);
 * -------------------------------------------------------
 * This function updates the vertical predicted velocity field with:
 *
 * 1) Old nonhydrostatic pressure gradient with theta method
 * 2) Horizontal and vertical advection of vertical momentum with AB2
 * 3) Horizontal laminar+turbulent diffusion of vertical momentum with AB2
 * 4) Vertical laminar+turbulent diffusion of vertical momentum with theta method
 *
 * Cn_W contains the Adams-Bashforth terms at time step n-1.
 * If wetting and drying is employed, no advection is computed in 
 * the upper cell.
 *
 */
static void WPredictor(gridT *grid, physT *phys, propT *prop,
		       int myproc, int numprocs, MPI_Comm comm) {
  int i, iptr, j, jptr, k, ne, nf, nc, nc1, nc2, kmin;
  REAL fab, sum, *a, *b, *c;
  int lagra_cell, lagra_layer;
  REAL  xs[2],zs,us[2],al[3],vtx[3],vty[3];
  
  a = phys->a;
  b = phys->b;
  c = phys->c;

  if(prop->n==1) {
    fab=1;
    for(i=0;i<grid->Nc;i++)
      for(k=0;k<grid->Nk[i];k++)
	phys->Cn_W[i][k]=0;
  } else
    fab=1.5;

  // Add on the nonhydrostatic pressure gradient from the previous time
  // step to compute the source term for the tridiagonal inversion.
  if(prop->nonlinear==3)
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr]; 
      
      if(grid->ctop[i]<grid->Nk[i]){
        for(k=grid->ctop[i];k<grid->Nk[i];k++) {
          phys->wtmp[i][k]=(1-fab)*phys->Cn_W[i][k];
          phys->Cn_W[i][k]=0;
      
        }
        for(k=grid->ctop[i]+1;k<grid->Nk[i];k++){
          if(grid->dzz[i][k-1]+grid->dzz[i][k]==0){
            printf("Error! Layer depth=0 at cell: %d, layer: %d \n",i,k);
            exit(1);
          }else
            phys->wtmp[i][k]-=2.0*prop->dt/(grid->dzz[i][k-1]+grid->dzz[i][k])*
	        (phys->q[i][k-1]-phys->q[i][k]);
      
	}
	if(grid->dzz[i][grid->ctop[i]]==0)
            printf("Error! Layer depth=0 at top layer cell: %d, layer: %d,h: %e, dv: %e \n",i,grid->ctop[i],phys->h[i],grid->dv[i]);
        else
          phys->wtmp[i][grid->ctop[i]]+=2.0*prop->dt/grid->dzz[i][grid->ctop[i]]*
                                        phys->q[i][grid->ctop[i]];
        
      }
    }
  else
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      if(grid->ctop[i]<grid->Nk[i]){
        for(k=grid->ctop[i];k<grid->Nk[i];k++) {
          phys->wtmp[i][k]=phys->w[i][k]+(1-fab)*phys->Cn_W[i][k];
          phys->Cn_W[i][k]=0;
        }

        for(k=grid->ctop[i]+1;k<grid->Nk[i];k++)
          phys->wtmp[i][k]-=2.0*prop->dt/(grid->dzz[i][k-1]+grid->dzz[i][k])*
	    (phys->q[i][k-1]-phys->q[i][k]);
        phys->wtmp[i][grid->ctop[i]]+=2.0*prop->dt/grid->dzz[i][grid->ctop[i]]*
          phys->q[i][grid->ctop[i]];
      }
    }
  
  for(i=0;i<grid->Nc;i++)
    for(k=0;k<grid->Nk[i];k++)
      phys->stmp[i][k]=0;

  // Compute Eulerian advection (nonlinear!=0)
  if(prop->nonlinear<3 && prop->nonlinear>0) {
    // Compute the w-component fluxes at the faces

    // Fluxes at boundary faces
    for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
      j = grid->edgep[jptr];

      for(k=grid->etop[j];k<grid->Nke[j];k++)
	phys->ut[j][k]=phys->boundary_w[jptr-grid->edgedist[2]][k]*grid->dzz[grid->grad[2*j]][k];
    }
    // Fluxes at boundary faces
    for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
      j = grid->edgep[jptr];

      for(k=grid->etop[j];k<grid->Nke[j];k++)
	phys->ut[j][k]=phys->boundary_w[jptr-grid->edgedist[2]][k]*grid->dzz[grid->grad[2*j]][k];
    }

    if(prop->nonlinear==1) // Upwind
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	j = grid->edgep[jptr];

	nc1 = grid->grad[2*j];
	nc2 = grid->grad[2*j+1];

	if(grid->ctop[nc1]>grid->ctop[nc2])
	  kmin = grid->ctop[nc1];
	else
	  kmin = grid->ctop[nc2];

	for(k=0;k<kmin;k++)
	  phys->ut[j][k]=0;

	for(k=kmin;k<grid->Nke[j];k++) {
	  if(phys->u[j][k]>0)
	    nc = nc2;
	  else
	    nc = nc1;

	  phys->ut[j][k]=0.5*(phys->w[nc][k]+phys->w[nc][k+1])*grid->dzz[nc][k];
	}
      }
    else // Central
      for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
	j = grid->edgep[jptr];

	nc1 = grid->grad[2*j];
	nc2 = grid->grad[2*j+1];

	if(grid->ctop[nc1]>grid->ctop[nc2])
	  kmin = grid->ctop[nc1];
	else
	  kmin = grid->ctop[nc2];

	for(k=0;k<kmin;k++)
	  phys->ut[j][k]=0;
	for(k=kmin;k<grid->Nke[j];k++)
	  phys->ut[j][k]=0.25*(InterpToFace(j,k,phys->w,phys->u,grid)+
			       InterpToFace(j,k+1,phys->w,phys->u,grid))*(grid->dzz[nc1][k]+grid->dzz[nc2][k]);
      }

    for(i=0;i<grid->Nc;i++) {

      for(nf=0;nf<NFACES;nf++) {

	ne = grid->face[i*NFACES+nf];

	for(k=grid->ctop[i]+1;k<grid->Nk[i];k++)
	  phys->stmp[i][k]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	    (grid->Ac[i]*grid->dzz[i][k]);

	// Top cell is filled with momentum from neighboring cells
	for(k=grid->etop[ne];k<=grid->ctop[i];k++)
	  phys->stmp[i][grid->ctop[i]]+=phys->ut[ne][k]*phys->u[ne][k]*grid->df[ne]*grid->normal[i*NFACES+nf]/
	    (grid->Ac[i]*grid->dzz[i][grid->ctop[i]]);
      }

      // Vertical advection
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
	phys->stmp[i][k]+=(pow(phys->w[i][k],2)-pow(phys->w[i][k+1],2))/grid->dzz[i][k];
      }
      // Top cell
      phys->stmp[i][grid->ctop[i]]-=pow(phys->w[i][grid->ctop[i]+1],2)/grid->dzz[i][grid->ctop[i]];
    }

    // Check to make sure integrated fluxes are 0 for conservation
    if(WARNING && prop->CdW==0 && prop->nu_H==0) {
      sum=0;
      for(i=0;i<grid->Nc;i++) {
	for(k=grid->ctop[i];k<grid->Nk[i];k++)
	  sum+=grid->Ac[i]*phys->stmp[i][k]*grid->dzz[i][k];
      }
      if(fabs(sum)>CONSERVED) printf("Warning, not W-momentum conservative!\n");
    }
  }
  
  if (prop->nonlinear==3){
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	if(grid->dzz[i][k]>0){
  	  xs[0]=grid->xv[i];
          xs[1]=grid->yv[i];
          zs=0;

	  LagraTracing(grid, phys, prop, i, k, &(lagra_cell), &(lagra_layer), xs,&zs,myproc);

  	  if(lagra_layer<grid->Nk[lagra_cell]&&grid->dzz[lagra_cell][lagra_layer]>0)
  	    phys->wtmp[i][k]+=(phys->w[lagra_cell][lagra_layer+1]*zs+phys->w[lagra_cell][lagra_layer]*(grid->dzz[lagra_cell][lagra_layer]-zs))
  	                      /grid->dzz[lagra_cell][lagra_layer];
          else
            printf("Error! after w tracing proc: %d, cell %d, layer: %d, dzz: %e, Nk: %d,h: %e, dv: %e  \n", 
            myproc,lagra_cell,lagra_layer, grid->dzz[lagra_cell][lagra_layer],grid->Nk[lagra_cell],phys->h[i],grid->dv[i]);
	}
	else{
	  printf("Error! proc: %d,Cell: %d, Layer: %d, dzz=%e. \n",myproc,i,k,grid->dzz[i][k]);
	  exit(0);
	}
      }
    }
  }
  
  // Add horizontal diffusion to stmp
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];
    
    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
    if(grid->ctop[nc1]>grid->ctop[nc2])
      kmin = grid->ctop[nc1];
    else
      kmin = grid->ctop[nc2];

    for(k=kmin;k<grid->Nke[j];k++) {
      a[k]=.5*prop->nu_H*(phys->w[nc2][k]-phys->w[nc1][k]+phys->w[nc2][k+1]-phys->w[nc1][k+1])*grid->df[j]/grid->dg[j];
      phys->stmp[nc1][k]-=a[k]/grid->Ac[nc1];  //Later Cn_W-=stmp..., so minus here. 
      phys->stmp[nc2][k]+=a[k]/grid->Ac[nc2];
    }
    for(k=grid->Nke[j];k<grid->Nk[nc1];k++) 
      phys->stmp[nc1][k]+=0.25*prop->CdW*fabs(phys->w[nc1][k]+phys->w[nc1][k+1])*
	(phys->w[nc1][k]+phys->w[nc1][k+1])*grid->df[j]/grid->Ac[nc1];
    for(k=grid->Nke[j];k<grid->Nk[nc2];k++) 
      phys->stmp[nc2][k]+=0.25*prop->CdW*fabs(phys->w[nc2][k]+phys->w[nc2][k+1])*
	(phys->w[nc2][k]+phys->w[nc2][k+1])*grid->df[j]/grid->Ac[nc2];
  }

  //Now use the cell-centered advection terms to update the advection at the faces
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr]; 
    
    for(k=grid->ctop[i]+1;k<grid->Nk[i];k++){
      if(grid->dzz[i][k-1]+grid->dzz[i][k]==0)
            printf("Error! Layer depth=0 at cell: %d, layer: %d,h: %e, dv: %e ctop: %d, Nk: %d\n",
              i,k,phys->h[i],grid->dv[i],grid->ctop[i],grid->Nk[i]); 
      else
        phys->Cn_W[i][k]-=prop->dt*(grid->dzz[i][k-1]*phys->stmp[i][k-1]+grid->dzz[i][k]*phys->stmp[i][k])/
	                  (grid->dzz[i][k-1]+grid->dzz[i][k]);
    }
    
    // Top flux advection consists only of top cell
    k=grid->ctop[i];
    if(k<grid->Nk[i])
      phys->Cn_W[i][k]-=prop->dt*phys->stmp[i][k];
  }
 
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr]; 
      for(k=grid->ctop[i];k<grid->Nk[i];k++){ 
        phys->wtmp[i][k]+=fab*phys->Cn_W[i][k];

      }
    }

  // wtmp now contains the right hand side without the vertical diffusion terms.  Now we
  // add the vertical diffusion terms to the explicit side and invert the tridiagonal for
  // vertical diffusion (only if grid->Nk[i]-grid->ctop[i]>=2)
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr]; 

    if(grid->Nk[i]-grid->ctop[i]>1) {
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
	a[k] = 2*(prop->nu+phys->nu_tv[i][k-1])/grid->dzz[i][k-1]/(grid->dzz[i][k]+grid->dzz[i][k-1]);
	b[k] = 2*(prop->nu+phys->nu_tv[i][k])/grid->dzz[i][k]/(grid->dzz[i][k]+grid->dzz[i][k-1]);
      }
      if(grid->dzz[i][grid->ctop[i]]>0){
        b[grid->ctop[i]]=(prop->nu+phys->nu_tv[i][grid->ctop[i]])/(grid->dzz[i][grid->ctop[i]]*grid->dzz[i][grid->ctop[i]]);
        a[grid->ctop[i]]=b[grid->ctop[i]];
      }
      else{
	printf("Error! proc: %d,Cell: %d, Layer: %d, dzz=%e. \n",myproc,i,grid->ctop[i],grid->dzz[i][grid->ctop[i]]);
	exit(1);
      }
      // Add on the explicit part of the vertical diffusion term
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++){ 
	phys->wtmp[i][k]+=prop->dt*(1-prop->theta)*(a[k]*phys->w[i][k-1]
						    -(a[k]+b[k])*phys->w[i][k]
						    +b[k]*phys->w[i][k+1]);

      }
      phys->wtmp[i][grid->ctop[i]]+=prop->dt*(1-prop->theta)*(-(a[grid->ctop[i]]+b[grid->ctop[i]])*phys->w[i][grid->ctop[i]]
							      +(a[grid->ctop[i]]+b[grid->ctop[i]])*phys->w[i][grid->ctop[i]+1]);
					      
      
      // Now formulate the components of the tridiagonal inversion.
      // c is the diagonal entry, a is the lower diagonal, and b is the upper diagonal.
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	c[k]=1+prop->dt*prop->theta*(a[k]+b[k]);
	a[k]*=(-prop->dt*prop->theta);
	b[k]*=(-prop->dt*prop->theta);
      }
      b[grid->ctop[i]]+=a[grid->ctop[i]];

      TriSolve(&(a[grid->ctop[i]]),&(c[grid->ctop[i]]),&(b[grid->ctop[i]]),
	       &(phys->wtmp[i][grid->ctop[i]]),&(phys->w[i][grid->ctop[i]]),grid->Nk[i]-grid->ctop[i]);
    } else {
      for(k=grid->ctop[i];k<grid->Nk[i];k++)
	phys->w[i][k]=phys->wtmp[i][k];
    }
  }
  
}

/*
 * Function: Corrector
 * Usage: Corrector(qc,grid,phys,prop,myproc,numprocs,comm);
 * ---------------------------------------------------------
 * Correct the horizontal velocity field with the pressure correction.
 * Do not correct velocities for which D[j]==0 since these are boundary
 * cells.
 *
 * After correcting the horizontal velocity, update the total nonhydrostatic
 * pressure with the pressure correction.
 *
 */
static void Corrector(REAL **qc, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm) {

  int i, iptr, j, jptr, k;
  REAL tmp;
  // Correct the horizontal velocity only if this is not a boundary point.
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr]; 
    if(phys->D[j]!=0 && grid->etop[j]<grid->Nke[j]-1)
      for(k=grid->etop[j];k<grid->Nke[j];k++){
	tmp=prop->dt/grid->dg[j]*
	  (qc[grid->grad[2*j]][k]-qc[grid->grad[2*j+1]][k]);
	phys->u[j][k]-=tmp;

      }
  }

  // Correct the vertical velocity
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr]; 
    for(k=grid->ctop[i]+1;k<grid->Nk[i];k++){
      tmp=2.0*prop->dt/(grid->dzz[i][k-1]+grid->dzz[i][k])*
	(qc[i][k-1]-qc[i][k]);
	phys->w[i][k]-=tmp;

    }
    if(grid->ctop[i]<grid->Nk[i]){	
      tmp=2.0*prop->dt/grid->dzz[i][grid->ctop[i]]*
      qc[i][grid->ctop[i]];
      phys->w[i][grid->ctop[i]]+=tmp;

     }
  }

  // Update the pressure since qc is a pressure correction
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];

    if(grid->ctop[i]<grid->Nk[i]-1)
      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	phys->q[i][k]+=qc[i][k];
  }
}

/*
 * Function: ComputeQSource
 * Usage: ComputeQSource(src,grid,phys,prop,myproc,numprocs);
 * ----------------------------------------------------------
 * Compute the source term for the nonhydrostatic pressure by computing
 * the divergence of the predicted velocity field, which is in phys->u and
 * phys->w.  The upwind flux face heights are used in order to ensure
 * consistency with continuity.
 *
 */
static void ComputeQSource(REAL **src, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs) {
  
  int i, iptr, j, jptr, k, nf, ne, nc1, nc2;
  REAL *ap=phys->a, *am=phys->b;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    
    for(k=grid->ctop[i];k<grid->Nk[i];k++) 
      src[i][k] = grid->Ac[i]*(phys->w[i][k]-phys->w[i][k+1]);

    for(nf=0;nf<NFACES;nf++) {

      ne = grid->face[i*NFACES+nf];
      nc1 = grid->grad[2*ne];
      nc2 = grid->grad[2*ne+1];
      if(nc1==-1) nc1=nc2;
      if(nc2==-1) nc2=nc1;

      for(k=grid->ctop[i];k<grid->Nke[ne];k++) {
	ap[k] = 0.5*(phys->u[ne][k]+fabs(phys->u[ne][k]));
	am[k] = 0.5*(phys->u[ne][k]-fabs(phys->u[ne][k]));
      }
      for(k=grid->Nke[ne];k<grid->Nkc[ne];k++) {
	ap[k] = 0;
	am[k] = 0;
      }

      for(k=grid->ctop[i];k<grid->Nke[ne];k++) 
	src[i][k]+=(grid->dzz[nc2][k]*ap[k]+grid->dzz[nc1][k]*am[k])*
	  grid->normal[i*NFACES+nf]*grid->df[ne];
    }

    for(k=grid->ctop[i];k<grid->Nk[i];k++) 
      src[i][k]/=prop->dt;
  }

  // D[j] is used in OperatorQ, and it must be zero to ensure no gradient
  // at the hydrostatic faces.
  for(j=0;j<grid->Ne;j++) {
    phys->D[j]=grid->df[j]/grid->dg[j];
  }

  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];

    for(nf=0;nf<NFACES;nf++)
      phys->D[grid->face[i*NFACES+nf]]=0;
  }
}

/*
 * Function: CGSolveQ
 * Usage: CGSolveQ(q,src,c,grid,phys,prop,myproc,numprocs,comm);
 * -------------------------------------------------------------
 * Solve for the nonhydrostatic pressure with the preconditioned
 * conjugate gradient algorithm.
 *
 * The preconditioner stores the diagonal preconditioning elements in 
 * the temporary c array.
 *
 * This function replaces q with x and src with p.  phys->uc and phys->vc
 * are used as temporary arrays as well to store z and r.
 *
 */
static void CGSolveQ(REAL **q, REAL **src, REAL **c, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm) {

  int i, iptr, k, n, niters;

  REAL **x, **r, **rtmp, **p, **z, mu, nu, alpha, alpha0, eps, eps0;

  z = phys->stmp2;
  x = q;
  r = phys->stmp3;
  rtmp = phys->uold;
  p = src;

  // Compute the preconditioner and the preconditioned solution
  // and send it to neighboring processors
  if(prop->qprecond==1) {
    ConditionQ(c,grid,phys,prop,myproc,comm);
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      
      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	p[i][k]/=c[i][k];
	x[i][k]*=c[i][k];
      }
    }
  }
  ISendRecvCellData3D(x,grid,myproc,comm);

  niters = prop->qmaxiters;

  // Create the coefficients for the operator
  QCoefficients(phys->wtmp,phys->qtmp,c,grid,phys,prop);
  
  // Initialization for CG
  if(prop->qprecond==1) OperatorQC(phys->wtmp,phys->qtmp,x,z,c,grid,phys,prop);
  else OperatorQ(phys->wtmp,x,z,c,grid,phys,prop);
   
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    
    for(k=grid->ctop[i];k<grid->Nk[i];k++) 
      r[i][k] = p[i][k]-z[i][k];
  }    
  if(prop->qprecond==2) {
    Preconditioner(r,rtmp,phys->wtmp,grid,phys,prop);
    
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	p[i][k] = rtmp[i][k];
    }
    alpha = alpha0 = InnerProduct3(r,rtmp,grid,myproc,numprocs,comm);
    
  } else {
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	p[i][k] = r[i][k];
    }
    alpha = alpha0 = InnerProduct3(r,r,grid,myproc,numprocs,comm);
  }
  if(!prop->resnorm) alpha0 = 1;

  if(prop->qprecond==2)
    eps=eps0=InnerProduct3(r,r,grid,myproc,numprocs,comm);
  else
    eps=eps0=alpha0;
  
  
  // Iterate until residual is less than prop->qepsilon
  for(n=0;n<niters && eps!=0;n++) {

    ISendRecvCellData3D(p,grid,myproc,comm);
    if(prop->qprecond==1) OperatorQC(phys->wtmp,phys->qtmp,p,z,c,grid,phys,prop);
    else OperatorQ(phys->wtmp,p,z,c,grid,phys,prop);

    mu = 1/alpha;
    nu = alpha/InnerProduct3(p,z,grid,myproc,numprocs,comm);

    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	x[i][k] += nu*p[i][k];
	r[i][k] -= nu*z[i][k];
      }
    }
    if(prop->qprecond==2) {
      Preconditioner(r,rtmp,phys->wtmp,grid,phys,prop);
      alpha = InnerProduct3(r,rtmp,grid,myproc,numprocs,comm);
      mu*=alpha;
      for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
	i = grid->cellp[iptr];

	for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	  p[i][k] = rtmp[i][k] + mu*p[i][k];
      }
    } else {
      alpha = InnerProduct3(r,r,grid,myproc,numprocs,comm);
      mu*=alpha;
      for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
	i = grid->cellp[iptr];

	for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	  p[i][k] = r[i][k] + mu*p[i][k];
      }
    }

    if(prop->qprecond==2)
      eps=InnerProduct3(r,r,grid,myproc,numprocs,comm);
    else
      eps=alpha;

    
    if(VERBOSE>3) printf("CGSolve Pressure Iteration: %d, resid=%e, proc=%d\n",n,sqrt(eps/eps0),myproc);
    if(sqrt(eps/eps0)<prop->qepsilon) 
      break;
  }
  if(myproc==0 && VERBOSE>2) 
    if(eps==0)
      printf("Warning...Time step %d, norm of pressure source is 0.\n",prop->n);
    else
      if(n==niters)  printf("Warning... Time step %d, Pressure iteration not converging after %d steps! RES=%e > %.2e\n",
			    prop->n,n,sqrt(eps/eps0),prop->qepsilon);
      else printf("Time step %d, CGSolve pressure converged after %d iterations, res=%e < %.2e\n",
		  prop->n,n,sqrt(eps/eps0),prop->qepsilon);

  // Rescale the preconditioned solution 
  if(prop->qprecond==1) {
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      
      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	x[i][k]/=c[i][k];
    }
  }
  
  // Send the solution to the neighboring processors
  ISendRecvCellData3D(x,grid,myproc,comm);
}

/*
 * Function: EddyViscosity
 * Usage: EddyViscosity(grid,phys,prop);
 * -------------------------------------
 * This function is used to compute the eddy viscosity, the
 * shear stresses, and the drag coefficients at the upper and lower
 * boundaries.
 *
 */
static void EddyViscosity(gridT *grid, physT *phys, sediT *sedi, propT *prop, MPI_Comm comm, int myproc)
{
  int i, k;

  if(prop->turbmodel) 
    my25(grid,phys, sedi, prop,phys->qT,phys->lT,phys->Cn_q,phys->Cn_l,phys->nu_tv,phys->kappa_tv,comm,myproc);
  for(i=0;i<grid->Nc;i++)
    for(k=grid->ctop[i];k<grid->Nk[i];k++){
      if(phys->nu_tv[i][k]!=phys->nu_tv[i][k] || phys->nu_tv[i][k]<1e-3)
	phys->nu_tv[i][k] = 1e-3;
    }
}

/*
 * Function: UPredictor 
 * Usage: UPredictor(grid,phys,prop,myproc,numprocs,comm);
 * -------------------------------------------------------
 * Predictor step for the horizontal velocity field.  This function
 * computes the free surface using the theta method and then uses it to update the predicted
 * velocity field in the absence of the nonhydrostatic pressure.
 *
 * Upon entry, phys->utmp contains the right hand side of the u-momentum equation
 *
 */
static void UPredictor(gridT *grid, physT *phys, 
		      propT *prop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, iptr, j, jptr, ne, nf, nf1, normal, nc1, nc2,nc, k;
  REAL sum, dt=prop->dt, theta=prop->theta, fluxheight, h0, boundary_flag;
  REAL *a, *b, *c, *d, *e1, **E, *a0, *b0, *c0, *d0, ustar[grid->Nkmax];   
  REAL temp,temp2;

  a = phys->a;
  b = phys->b;
  c = phys->c;
  d = phys->d;
  e1 = phys->ap;
  E = phys->ut;

  a0 = phys->am;
  b0 = phys->bp;
  c0 = phys->bm;

  // Set D[j] = 0 
  for(i=0;i<grid->Nc;i++) 
    for(k=0;k<grid->Nk[i]+1;k++) 
      phys->wtmp2[i][k]=phys->w[i][k];


  for(j=0;j<grid->Ne;j++) {
    phys->D[j]=0;
    for(k=0;k<grid->Nke[j];k++)
      phys->utmp2[j][k]=phys->u[j][k];   
  }



  // phys->u contains the velocity specified at the open boundaries
  // It is also the velocity at time step n.
  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];

    for(k=grid->etop[j];k<grid->Nke[j];k++) 
      phys->utmp[j][k]=phys->u[j][k];    
  }
  


  // Update the velocity in the interior nodes with the old free-surface gradient
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {

    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];

    // Add the explicit part of the free-surface to create U**.
    if (grid->etop[j]<grid->Nke[j]){
      //If one cell is dry 
      /*if(grid->ctop[nc1]==grid->Nk[nc1]){
      	sum=0;
        for(k=grid->etop[j];k<grid->Nke[j];k++){ 
          sum+=grid->dzz[nc2][k]/2.0;
          phys->utmp[j][k]+=GRAV*dt*sum/grid->dg[j];
          sum+=grid->dzz[nc2][k]/2.0;
          printf("edge: %d, layer: %d, utmp: %e \n",j,k,phys->utmp[j][k]);
        }
      }
      else if(grid->ctop[nc2]==grid->Nk[nc2]){
      	sum=0;
        for(k=grid->etop[j];k<grid->Nke[j];k++){ 
          sum+=grid->dzz[nc1][k]/2.0;
          phys->utmp[j][k]-=GRAV*dt*sum/grid->dg[j];
          sum+=grid->dzz[nc1][k]/2.0;
          printf("edge: %d, layer: %d, utmp: %e \n",j,k,phys->utmp[j][k]);
        }
      }
      else*/
        for(k=grid->etop[j];k<grid->Nke[j];k++){ 

          phys->utmp[j][k]-=GRAV*(1-theta)*dt*(phys->h[nc1]-phys->h[nc2])/grid->dg[j];

        }
	
    
    } 

 
  }

  // Update the boundary faces with the linearized free-surface gradient at the boundary
  // i.e. using the radiative condition, dh/dn = 1/c dh/dt
  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    i = grid->grad[2*j];
    boundary_flag=phys->boundary_flag[jptr-grid->edgedist[2]];
    for(k=grid->etop[j];k<grid->Nke[j];k++) 
      phys->utmp[j][k]+=(-2.0*GRAV*(1-boundary_flag)*(1-theta)*dt/grid->dg[j]
			 +boundary_flag*sqrt(GRAV/(grid->dv[i]+phys->h[i])))*phys->h[i]
	+2.0*GRAV*(1-boundary_flag)*dt/grid->dg[j]*phys->boundary_h[jptr-grid->edgedist[2]];
  }


  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];
    if(grid->etop[j]<grid->Nke[j]){
      nc1 = grid->grad[2*j];
      nc2 = grid->grad[2*j+1];
      
      if(nc1==-1)
        nc1=nc2;
      if(nc2==-1)
        nc2=nc1;
      h0 = 0.5*(grid->dzz[nc1][grid->etop[j]]+grid->dzz[nc2][grid->etop[j]]);
      // Add the shear stress from the top cell
      if (h0 < 0.2)
	h0 = 0.2;

      //Modified by YJChou to apply realistic wind field
  
     //      phys->tau_T[j] = phys->CdT[j]*fabs(wave->Uwind[j]-phys->u[j][grid->etop[j]])
      //	*(wave->Uwind[j]-phys->u[j][grid->etop[j]]);
      //phys->utmp[j][grid->etop[j]]+=dt*phys->tau_T[j]/h0;
      

      // Create the tridiagonal entries and formulate U***
      if(!(grid->dzz[nc1][grid->etop[j]]==0 && grid->dzz[nc2][grid->etop[j]]==0)) {
      
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  a[k]=0;
	  b[k]=0;
	  c[k]=0;
	  d[k]=0;
        }
        
        // Vertical eddy-viscosity interpolated to faces since it is stored
        // at cell-centers.
        for(k=grid->etop[j]+1;k<grid->Nke[j];k++) 
	  c[k]=0.25*(phys->nu_tv[nc1][k-1]+phys->nu_tv[nc2][k-1]+
	  	   phys->nu_tv[nc1][k]+phys->nu_tv[nc2][k]);
        
        // Coefficients for the viscous terms.  Face heights are taken as
        // the average of the face heights on either side of the face (not upwinded).
        for(k=grid->etop[j]+1;k<grid->Nke[j];k++){
	
          a[k]=2.0*(prop->nu+c[k])/(0.25*(grid->dzz[nc1][k]+grid->dzz[nc2][k])*
	  			  (grid->dzz[nc1][k-1]+grid->dzz[nc2][k-1]+
	  			   grid->dzz[nc1][k]+grid->dzz[nc2][k]));
        }
        
        for(k=grid->etop[j];k<grid->Nke[j]-1;k++) {
	  b[k]=2.0*(prop->nu+c[k+1])/(0.25*(grid->dzz[nc1][k]+grid->dzz[nc2][k])*
	  			    (grid->dzz[nc1][k]+grid->dzz[nc2][k]+
	  			     grid->dzz[nc1][k+1]+grid->dzz[nc2][k+1]));	 
        }

        if(grid->Nke[j]-grid->etop[j]>1) {
      
	  // Explicit part of the viscous term
	  for(k=grid->etop[j]+1;k<grid->Nke[j]-1;k++)
	    phys->utmp[j][k]+=dt*(1-theta)*(a[k]*phys->u[j][k-1]-
	  				  (a[k]+b[k])*phys->u[j][k]+
	  				  b[k]*phys->u[j][k+1]);
	  
	  //Modified by YJChou to accound for realistic wind field
	  // Top cell
	  //	  if(wprop->wind_shear)	  
	  // phys->utmp[j][grid->etop[j]]+=dt*(1-theta)*(-(b[grid->etop[j]]*phys->u[j][grid->etop[j]]
	  //					+2.0*phys->CdT[j]*
	  //					fabs(wave->Uwind[j]-phys->u[j][grid->etop[j]])/
	  //					h0*
	  //					(wave->Uwind[j]-phys->u[j][grid->etop[j]])							
	  //					+b[grid->etop[j]]*phys->u[j][grid->etop[j]+1]));
	  //else
	 	  phys->utmp[j][grid->etop[j]]+=dt*(1-theta)*(-(b[grid->etop[j]]+2.0*phys->CdT[j]*
	  	  					      fabs(phys->u[j][grid->etop[j]])/
	  					      (grid->dzz[nc1][grid->etop[j]]+
	  					       grid->dzz[nc2][grid->etop[j]]))*
	  					    phys->u[j][grid->etop[j]]
	  					      +b[grid->etop[j]]*phys->u[j][grid->etop[j]+1]);

	  // Bottom cell
	  phys->utmp[j][grid->Nke[j]-1]+=dt*(1-theta)*(a[grid->Nke[j]-1]*phys->u[j][grid->Nke[j]-2]-
	  					     (a[grid->Nke[j]-1]+2.0*phys->CdB[j]*
	  					      fabs(phys->u[j][grid->Nke[j]-1])/
	  					      (grid->dzz[nc1][grid->Nke[j]-1]+
	  					       grid->dzz[nc2][grid->Nke[j]-1]))*
	  					     phys->u[j][grid->Nke[j]-1]);
        
        } else 
	  phys->utmp[j][grid->etop[j]]-=2.0*dt*(1-theta)*(phys->CdB[j]+phys->CdT[j])/
	    (grid->dzz[nc1][grid->etop[j]]+grid->dzz[nc2][grid->etop[j]])*
	    fabs(phys->u[j][grid->etop[j]])*phys->u[j][grid->etop[j]];
      
        // Now set up the coefficients for the tridiagonal inversion for the
        // implicit part.  These are given from the arrays above in the discrete operator
        // d^2U/dz^2 = -theta dt a_k U_{k-1} + (1+theta dt (a_k+b_k)) U_k - theta dt b_k U_{k+1}
      

        

        // Right hand side U** is given by d[k] here.
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  e1[k]=1.0;
	  d[k]=phys->utmp[j][k];
        }
        
        if(grid->Nke[j]-grid->etop[j]>1) {
	  // Top cells
	  c[grid->etop[j]]=-theta*dt*b[grid->etop[j]];
	  
	  // Modified by YJChou
	  //if(wprop->wind_shear){
	  // b[grid->etop[j]]=1.0+theta*dt*(b[grid->etop[j]]-
	  //			   2.0*phys->CdT[j]*fabs(wave->Uwind[j]-phys->u[j][grid->etop[j]])
	  //			   /h0);	 					 
	  // d[grid->etop[j]] += theta*dt*2.0*phys->CdT[j]
	  //   *fabs(wave->Uwind[j]-phys->u[j][grid->etop[j]])*wave->Uwind[j]
	  //				   /h0;
	  //}else
	   b[grid->etop[j]]=1.0+theta*dt*(b[grid->etop[j]]+
	  			   2.0*phys->CdT[j]*fabs(phys->u[j][grid->etop[j]])/
	  			   (grid->dzz[nc1][grid->etop[j]]+
	  				    grid->dzz[nc2][grid->etop[j]]));
      
	  //////////////////////////////////////////////////////
	  a[grid->etop[j]]=0;	  

	  // Bottom cell
	  c[grid->Nke[j]-1]=0;
	  b[grid->Nke[j]-1]=1.0+theta*dt*(a[grid->Nke[j]-1]+
	  				2.0*phys->CdB[j]*fabs(phys->u[j][grid->Nke[j]-1])/
	  				(grid->dzz[nc1][grid->Nke[j]-1]+
	  				 grid->dzz[nc2][grid->Nke[j]-1]));
	  a[grid->Nke[j]-1]=-theta*dt*a[grid->Nke[j]-1];
        
	  // Interior cells
	  for(k=grid->etop[j]+1;k<grid->Nke[j]-1;k++) {
	    c[k]=-theta*dt*b[k];
	    b[k]=1.0+theta*dt*(a[k]+b[k]);
	    a[k]=-theta*dt*a[k];
	  }
        } else {
	  b[grid->etop[j]]=1.0+2.0*theta*dt*fabs(phys->u[j][grid->etop[j]])/
	    (grid->dzz[nc1][grid->etop[j]]+grid->dzz[nc2][grid->etop[j]])*
	    (phys->CdB[j]+phys->CdT[j]);
        }	  
      
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  if(grid->dzz[nc1][k]==0 && grid->dzz[nc2][k]==0) {
	    printf("Exiting because dzz[%d][%d]=%f or dzz[%d][%d]=%f\n",
	  	 nc1,k,grid->dzz[nc1][k],nc2,k,grid->dzz[nc2][k]);
	    exit(0);
	  }
	  if(a[k]!=a[k]) printf("a[%d] problems, dzz[%d][%d]=%f\n",k,j,k,grid->dzz[j][k]);
	  if(b[k]!=b[k] || b[k]==0) printf("b[%d] problems\n",k);
	  if(c[k]!=c[k]) printf("c[%d] problems\n",k);
        }

        // Now utmp will have U*** in it, which is given by A^{-1}U**, and E will have
        // A^{-1}e1, where e1 = [1,1,1,1,1,...,1]^T 
        // Store the tridiagonals so they can be used twice (TriSolve alters the values
        // of the elements in the diagonals!!!
        for(k=0;k<grid->Nke[j];k++) {
	  a0[k]=a[k];
	  b0[k]=b[k];
	  c0[k]=c[k];
        }

        if(grid->Nke[j]-grid->etop[j]>1) {
	  TriSolve(&(a[grid->etop[j]]),&(b[grid->etop[j]]),&(c[grid->etop[j]]),
	  	 &(d[grid->etop[j]]),&(phys->utmp[j][grid->etop[j]]),grid->Nke[j]-grid->etop[j]);
	  TriSolve(&(a0[grid->etop[j]]),&(b0[grid->etop[j]]),&(c0[grid->etop[j]]),
	  	 &(e1[grid->etop[j]]),&(E[j][grid->etop[j]]),grid->Nke[j]-grid->etop[j]);
        } else {
	  phys->utmp[j][grid->etop[j]]/=b[grid->etop[j]];
	  E[j][grid->etop[j]]=1.0/b[grid->etop[j]];
        }
      
        // Now vertically integrate E to create the vertically integrated flux-face
        // values that comprise the coefficients of the free-surface solver.  This
        // will create the D vector, where D=DZ^T E (which should be given by the
        // depth when there is no viscosity.
        phys->D[j]=0;
        for(k=grid->etop[j];k<grid->Nke[j];k++) {
	  fluxheight=UpWind(phys->u[j][k],grid->dzz[nc1][k],grid->dzz[nc2][k]);
      
	  phys->D[j]+=fluxheight*E[j][k];
        }
      } 
    }  //End wet cell if

    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      if(phys->utmp[j][k]!=phys->utmp[j][k]) {
	printf("Proc:%d Error in function at j%d k=%d utmp=%f\n",myproc,j,k,phys->utmp[j][k]);
	printf("etop=%d mark=%d a=%f b=%f c=%f\n",grid->etop[j],grid->mark[j],a[k],b[k],c[k]);
      }
    }
  } //End inner edge loop
  
  // bing changed to give more info
  for(j=0;j<grid->Ne;j++) 
    for(k=grid->etop[j];k<grid->Nke[j];k++) 
      if(phys->utmp[j][k]!=phys->utmp[j][k]) {
	printf(" Proc:%d Error in function Predictor at j=%d k=%d (U***=nan)\n",myproc,j,k);
	printf(" xe=%1.6e, ye=%1.6e, etop=%d, mark=%d\n",grid->xe[j],grid->ye[j],grid->etop[j],grid->mark[j]);
	exit(1);
      }

  // So far we have U*** and D.  Now we need to create h* in htmp.   This
  // will comprise the source term for the free-surface solver.  Before we
  // do this we need to set the new velocity at the open boundary faces and
  // place them into utmp.  These need the velocity from the old time step uold
  // As well as the current vectors.
  OpenBoundaryFluxes(phys->q,phys->utmp,phys->utmp2,grid,phys,prop);
  
  // added for conservation  Oct 2006, Gang
  ISendRecvEdgeData3D(phys->utmp, grid, myproc, comm);

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];

    sum = 0;
    for(nf=0;nf<NFACES;nf++) {

      ne = grid->face[i*NFACES+nf];

      normal = grid->normal[i*NFACES+nf];
      nc1 = grid->grad[2*ne];
      nc2 = grid->grad[2*ne+1];
      if(nc1==-1) nc1=nc2;
      if(nc2==-1) nc2=nc1;

      for(k=grid->etop[ne];k<grid->Nke[ne];k++) {
        if(phys->u[ne][k]>=0)
          nc=nc2;
        else
          nc=nc1;
        fluxheight=grid->dzz[nc][k];
        temp=((1-theta)*phys->u[ne][k]+theta*phys->utmp[ne][k])*fluxheight*grid->df[ne]*normal;

	sum+=temp;
      }
    }

    phys->htmp[i]=phys->h[i]-dt/grid->Ac[i]*sum;
  }
  
  for(i=0; i<grid->Nc; i++){
    phys->dhdt[i]=phys->h[i];
  }

  // Now we have the required components for the CG solver for the free-surface:
  // h^{n+1} - g*(theta*dt)^2/Ac * Sum_{faces} D_{face} dh^{n+1}/dn df N = htmp
  // L(h) = b
  // L(h) = h + 1/Ac * Sum_{faces} D_{face} dh^{n+1}/dn N
  // b = htmp
  // As the initial guess let h^{n+1} = h^n, so just leave it as it is to
  // begin the solver.
  if(prop->cgsolver==0)
    GSSolve(grid,phys,prop,myproc,numprocs,comm);
  else if(prop->cgsolver==1)
    CGSolve(grid,phys,prop,myproc,numprocs,comm);
    //BiCGSolveN(1, 1, grid,phys,prop,myproc,numprocs,comm);
  for(i=0; i<grid->Nc; i++){
    phys->dhdt[i]=(phys->h[i]-phys->dhdt[i])/dt;
  }

  

  // Add on the implicit barotropic term to obtain the hydrostatic horizontal velocity field.
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];
      for(k=grid->etop[j];k<grid->Nke[j];k++){
        phys->u[j][k]=phys->utmp[j][k]-GRAV*theta*dt*E[j][k]*
  	(phys->h[nc1]-phys->h[nc2])/grid->dg[j];

      }
    //}
    if(grid->etop[j]==grid->Nke[j]-1 && grid->dzz[nc1][grid->etop[j]]==0 &&
       grid->dzz[nc2][grid->etop[j]]==0)
      phys->u[j][grid->etop[j]]=0;
  }

  // Set the flux values at boundary cells if specified (marker=4)
  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    i = grid->grad[2*j];
    boundary_flag = phys->boundary_flag[jptr-grid->edgedist[2]];
    for(k=grid->etop[j];k<grid->Nke[j];k++) 
      phys->u[j][k]=phys->utmp[j][k]-(2.0*GRAV*(1-boundary_flag)*theta*dt/grid->dg[j]+
				      boundary_flag*sqrt(GRAV/(grid->dv[i]+phys->h[i])))*E[j][k]*phys->h[i];
  }

  // Now update the vertical grid spacing with the new free surface.
  UpdateDZ(grid,phys,0);
  
 


  // Use the new free surface to add the implicit part of the free-surface
  // pressure gradient to the horizontal momentum.
  for(jptr=grid->edgedist[0];jptr<grid->edgedist[1];jptr++) {
    j = grid->edgep[jptr];

    nc1 = grid->grad[2*j];
    nc2 = grid->grad[2*j+1];

    if(grid->etop[j]>grid->etopold[j]) 
      for(k=0;k<grid->etop[j];k++)
	phys->u[j][k]=0;
    else
      for(k=grid->etopold[j]-1;k>=grid->etop[j];k--)
        if(k+1<grid->Nke[j])
          phys->u[j][k]= phys->u[j][k+1];
        else  //Dry edge at the previous step
          phys->u[j][k]=-GRAV*dt*(phys->h[nc1]-phys->h[nc2])/grid->dg[j];
 
  }

  // Set the flux values at boundary cells if specified (marker=4)
  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    i = grid->grad[2*j];
    boundary_flag = phys->boundary_flag[jptr-grid->edgedist[2]];
    if(grid->etop[j]>grid->etopold[j]) 
      for(k=0;k<grid->etop[j];k++)
	phys->u[j][k]=0;
    else 
      for(k=grid->etop[j];k<grid->etopold[j];k++) 
	phys->u[j][k]=phys->utmp[j][k]-(2.0*GRAV*(1-boundary_flag)*theta*dt/grid->dg[j]+
					boundary_flag*sqrt(GRAV/(grid->dv[i]+phys->h[i])))*phys->h[i];
  }

  // Set the flux values at the open boundary (marker=2).  These
  // were set to utmp previously in OpenBoundaryFluxes.
  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
    j = grid->edgep[jptr];

    for(k=grid->etop[j];k<grid->Nke[j];k++) {
      phys->u[j][k] = phys->utmp[j][k];

    }
  }
  
  // Now set the fluxes at the free-surface boundary by assuming dw/dz = 0
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
    for(nf=0;nf<NFACES;nf++) {
      ne = grid->face[i*NFACES+nf];
      if(grid->mark[ne]==3) {
	for(k=grid->etop[ne];k<grid->Nke[ne];k++) {
	  phys->u[ne][k] = 0;
	  sum=0;
	  for(nf1=0;nf1<NFACES;nf1++)
	    sum+=phys->u[grid->face[i*NFACES+nf1]][k]*grid->df[grid->face[i*NFACES+nf1]]*grid->normal[i*NFACES+nf1];
	  phys->u[ne][k]=-sum/grid->df[ne]/grid->normal[i*NFACES+nf];
	}
      }
    }
  }
}

/*
 * Function: CGSolve
 * Usage: CGSolve(grid,phys,prop,myproc,numprocs,comm);
 * ----------------------------------------------------
 * Solve the free surface equation using the conjugate gradient algorithm.
 *
 * The source term upon entry is in phys->htmp, which is placed into p, and
 * the free surface upon entry is in phys->h, which is placed into x.
 *
 */
static void CGSolve(gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, iptr, n, niters;
  REAL *x, *r, *D, *p, *z, mu, nu, eps, eps0;

  z = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"CGSolve");
  x = phys->h;
  r = phys->hold;
  D = phys->D;
  p = phys->htmp;

  niters = prop->maxiters;

  // For the boundary term (marker of type 3):
  // 1) Need to set x to zero in the interior points, but
  //    leave it as is for the boundary points.
  // 2) Then set z=Ax and substract b = b-z so that
  //    the new problem is Ax=b with the boundary values
  //    on the right hand side acting as forcing terms.
  // 3) After b=b-z for the interior points, then need to
  //    set b=0 for the boundary points.
  ISendRecvCellData2D(x,grid,myproc,comm);
  OperatorH(x,z,grid,phys,prop);

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    p[i] = p[i] - z[i];
    r[i] = p[i];
  } 
  
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
    p[i] = 0;
  }    
  eps0 = eps = InnerProduct(r,r,grid,myproc,numprocs,comm);
  if(!prop->resnorm) eps0 = 1;
  
  for(n=0;n<niters && eps!=0;n++) {

    ISendRecvCellData2D(p,grid,myproc,comm);
    OperatorH(p,z,grid,phys,prop);
  
    mu = 1/eps;
    nu = eps/InnerProduct(p,z,grid,myproc,numprocs,comm);

    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      x[i] += nu*p[i];
      r[i] -= nu*z[i];
    }
    eps = InnerProduct(r,r,grid,myproc,numprocs,comm);
    mu*=eps;

    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      p[i] = r[i] + mu*p[i];
    }

    if(VERBOSE>3) printf("CGSolve Iteration: %d, resid=%e, proc=%d\n",n,sqrt(eps/eps0),myproc);
    if(sqrt(eps/eps0)<prop->epsilon) 
      break;
  }
  if(myproc==0 && VERBOSE>2) 
    if(eps==0)
      printf("Warning...Time step %d, norm of free-surface source is 0.\n",prop->n);
    else
      if(n==niters)  printf("Warning... Time step %d, Free-surface iteration not converging after %d steps! RES=%e > %.2e\n",
			    prop->n,n,sqrt(eps/eps0),prop->epsilon);
      else printf("Time step %d, CGSolve surface converged after %d iterations, res=%e < %.2e\n",
		  prop->n,n,sqrt(eps/eps0),prop->epsilon);

  ISendRecvCellData2D(x,grid,myproc,comm);
  SunFree(z,grid->Nc*sizeof(REAL),"CGSolve");
}






static void BiCGSolveN2(int m0, int n0, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, iptr, n, niters;
  REAL *x, *r, *p, *z, *r0, *v, *t, *s, eps, eps0;
  REAL rho0 = 1, rho, alpha = 1, omg=1, beta, tmp;

  z = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"BiCGSolveN");
  v = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"BiCGSolveN");
  s = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"BiCGSolveN");
  t = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"BiCGSolveN");
  r0 = (REAL *)SunMalloc(grid->Nc*sizeof(REAL),"BiCGSolveN");
  x = phys->h;
  r = phys->hold; 
  p = phys->htmp;
  

  niters = prop->maxiters;
  
  for(i = 0; i < grid->Nc; i++){
    z[i] = 0;
    v[i] = 0;
    s[i] = 0;
    t[i] = 0;
    r0[i] = 0;
  }


  // For the boundary term (marker of type 3):
  // 1) Need to set x to zero in the interior points, but
  //    leave it as is for the boundary points.
  // 2) Then set z=Ax and substract b = b-z so that
  //    the new problem is Ax=b with the boundary values
  //    on the right hand side acting as forcing terms.
  // 3) After b=b-z for the interior points, then need to
  //    set b=0 for the boundary points.
  ISendRecvCellData2D(x,grid,myproc,comm);
  OperatorH(x,z,grid,phys,prop);

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    p[i] = p[i] - z[i];           //initial residual p = b-Ax
    r[i] = p[i];
    r0[i] = r[i];
  } 
  
  for(iptr=grid->celldist[1];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
    p[i] = 0;
  }    
  eps0 = eps = InnerProduct(r,r,grid,myproc,numprocs,comm);
  if(!prop->resnorm) eps0 = 1;
  


  for(n=0;n<niters && eps!=0;n++) {

    rho = InnerProduct(r0,r,grid,myproc,numprocs,comm);
    beta = rho/rho0*alpha/omg;
    
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      p[i] = r[i]+beta*(p[i]-omg*v[i]);

    }

    ISendRecvCellData2D(p,grid,myproc,comm);
    OperatorH(p,v,grid,phys,prop);
    tmp = InnerProduct(r0,v,grid,myproc,numprocs,comm);
    tmp  =1/tmp;
    alpha = rho*tmp;
    
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      s[i] = r[i]-alpha*v[i];
    }
    
    ISendRecvCellData2D(s,grid,myproc,comm);
    OperatorH(s,t,grid,phys,prop);
     
    tmp = InnerProduct(t, t,grid,myproc,numprocs,comm);
    tmp = 1/tmp;
    omg = InnerProduct(t, s,grid,myproc,numprocs,comm)*tmp;

    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      x[i] += alpha*p[i] + omg*s[i];
      r[i] = s[i]-omg*t[i];
    }
    rho0 = rho;
    
    eps = InnerProduct(r,r,grid,myproc,numprocs,comm);

    if(VERBOSE>3) printf("CGSolve Iteration: %d, resid=%e, proc=%d\n",n,sqrt(eps/eps0),myproc);
    if(sqrt(eps/eps0)<prop->epsilon) 
      break;
  }
  if(myproc==0 && VERBOSE>2) 
    if(eps==0)
      printf("Warning...Time step %d, norm of free-surface source is 0.\n",prop->n);
    else
      if(n==niters)  printf("Warning... Time step %d, Free-surface iteration not converging after %d steps! RES=%e > %.2e\n",
			    prop->n,n,sqrt(eps/eps0),prop->epsilon);
      else printf("Time step %d, CGSolve surface converged after %d iterations, res=%e < %.2e\n",
		  prop->n,n,sqrt(eps/eps0),prop->epsilon);

  ISendRecvCellData2D(x,grid,myproc,comm);
  SunFree(z,grid->Nc*sizeof(REAL),"BiCGSolveN");
  SunFree(v,grid->Nc*sizeof(REAL),"BiCGSolveN");
  SunFree(s,grid->Nc*sizeof(REAL),"BiCGSolveN");
  SunFree(t,grid->Nc*sizeof(REAL),"BiCGSolveN");
  SunFree(r0,grid->Nc*sizeof(REAL),"BiCGSolveN");
}

/*
 * Function: InnerProduct
 * Usage: InnerProduct(x,y,grid,myproc,numprocs,comm);
 * ---------------------------------------------------
 * Compute the inner product of two one-dimensional arrays x and y.
 * Used for the CG method to solve for the free surface.
 *
 */
static REAL InnerProduct(REAL *x, REAL *y, gridT *grid, int myproc, int numprocs, MPI_Comm comm) {
  
  int i, iptr;
  REAL sum, mysum=0;
  
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    mysum+=x[i]*y[i];
  }
  MPI_Reduce(&mysum,&(sum),1,MPI_DOUBLE,MPI_SUM,0,comm);
  MPI_Bcast(&sum,1,MPI_DOUBLE,0,comm);

  return sum;
}  

/*
 * Function: InnerProduct3
 * Usage: InnerProduct3(x,y,grid,myproc,numprocs,comm);
 * ---------------------------------------------------
 * Compute the inner product of two two-dimensional arrays x and y.
 * Used for the CG method to solve for the nonhydrostatic pressure.
 *
 */
static REAL InnerProduct3(REAL **x, REAL **y, gridT *grid, int myproc, int numprocs, MPI_Comm comm) {
  
  int i, k, iptr;
  REAL sum, mysum=0;
  
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];

    for(k=grid->ctop[i];k<grid->Nk[i];k++)
      mysum+=x[i][k]*y[i][k];
  }
  MPI_Reduce(&mysum,&(sum),1,MPI_DOUBLE,MPI_SUM,0,comm);
  MPI_Bcast(&sum,1,MPI_DOUBLE,0,comm);

  return sum;
}  

/*
 * Function: OperatorH
 * Usage: OperatorH(x,y,grid,phys,prop);
 * -------------------------------------
 * Given a vector x, computes the left hand side of the free surface 
 * Poisson equation and places it into y with y = L(x).
 *
 */
static void OperatorH(REAL *x, REAL *y, gridT *grid, physT *phys, propT *prop) {
  
  int i, j, iptr, jptr, ne, nf;
  REAL tmp= GRAV*pow(prop->theta*prop->dt,2), h0, boundary_flag;
  int nc;
  
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      y[i] = x[i];
      for(nf=0;nf<NFACES;nf++){
	nc=grid->neigh[i*NFACES+nf]; 
	ne = grid->face[i*NFACES+nf];
	if(nc!=-1 && (grid->etop[ne]<grid->Nke[ne]))             
	  y[i]=y[i]+tmp*phys->D[ne]*grid->df[ne]/grid->dg[ne]*
	    (x[i]-x[nc])/grid->Ac[i];
 	 
      }
  }

  for(jptr=grid->edgedist[4];jptr<grid->edgedist[5];jptr++) {
    j = grid->edgep[jptr];

    i = grid->grad[2*j];
    boundary_flag = phys->boundary_flag[jptr-grid->edgedist[2]];
    y[i] += prop->dt*prop->theta*(2.0*GRAV*(1-boundary_flag)*prop->theta*prop->dt/grid->dg[j]
				  +boundary_flag*sqrt(GRAV/(grid->dv[i]+phys->boundary_h[jptr-grid->edgedist[2]])))*
      (grid->dv[i]+phys->boundary_h[jptr-grid->edgedist[2]])*grid->df[j]/grid->Ac[i]*x[i];
  }
}

/*
 * Function: OperatorQC
 * Usage: OperatorQC(coef,fcoef,x,y,c,grid,phys,prop);
 * ---------------------------------------------------
 * Given a vector x, computes the left hand side of the nonhydrostatic pressure
 * Poisson equation and places it into y with y = L(x) for the preconditioned
 * solver.
 *
 * The coef array contains coefficients for the vertical derivative terms in the operator
 * while the fcoef array contains coefficients for the horizontal derivative terms.  These
 * are computed before the iteration in QCoefficients. The array c stores the preconditioner.
 *
 */
static void OperatorQC(REAL **coef, REAL **fcoef, REAL **x, REAL **y, REAL **c, gridT *grid, physT *phys, propT *prop) {

  int i, iptr, k, ne, nf, nc, kmin, kmax;
  REAL *a = phys->a;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	y[i][k]=-x[i][k];

      for(nf=0;nf<NFACES;nf++) 
	if((nc=grid->neigh[i*NFACES+nf])!=-1) {

	    ne = grid->face[i*NFACES+nf];

	    if(grid->ctop[nc]>grid->ctop[i])
	      kmin = grid->ctop[nc];
	    else
	      kmin = grid->ctop[i];
	    
	    for(k=kmin;k<grid->Nke[ne];k++) 
	      y[i][k]+=x[nc][k]*fcoef[i*NFACES+nf][k];
	  }

      for(k=grid->ctop[i]+1;k<grid->Nk[i]-1;k++)
	y[i][k]+=coef[i][k]*x[i][k-1]+coef[i][k+1]*x[i][k+1];

      if(grid->ctop[i]<grid->Nk[i]-1) {
	// Top q=0 so q[i][grid->ctop[i]-1]=-q[i][grid->ctop[i]]
	k=grid->ctop[i];
	y[i][k]+=coef[i][k+1]*x[i][k+1];

	// Bottom dq/dz = 0 so q[i][grid->Nk[i]]=q[i][grid->Nk[i]-1]
	k=grid->Nk[i]-1;
	y[i][k]+=coef[i][k]*x[i][k-1];
      }
  }
}

/*
 * Function: QCoefficients
 * Usage: QCoefficients(coef,fcoef,c,grid,phys,prop);
 * --------------------------------------------------
 * Compute coefficients for the pressure-Poisson equation.  fcoef stores
 * coefficients at the vertical flux faces while coef stores coefficients
 * at the horizontal faces for vertical derivatives of q.
 *
 */
static void QCoefficients(REAL **coef, REAL **fcoef, REAL **c, gridT *grid, physT *phys, propT *prop) {

  int i, iptr, k, kmin, nf, nc, ne;

  if(prop->qprecond==1) 
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
      if(grid->ctop[i]<grid->Nk[i])
        coef[i][grid->ctop[i]]=grid->Ac[i]/grid->dzz[i][grid->ctop[i]]/c[i][grid->ctop[i]];
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) 
	coef[i][k] = 2*grid->Ac[i]/(grid->dzz[i][k]+grid->dzz[i][k-1])/(c[i][k]*c[i][k-1]);

      for(nf=0;nf<NFACES;nf++) 
	if((nc=grid->neigh[i*NFACES+nf])!=-1) {

	    ne = grid->face[i*NFACES+nf];

	    if(grid->ctop[nc]>grid->ctop[i])
	      kmin = grid->ctop[nc];
	    else
	      kmin = grid->ctop[i];
	    
	    for(k=kmin;k<grid->Nke[ne];k++) 
	      fcoef[i*NFACES+nf][k]=grid->dzz[i][k]*phys->D[ne]/(c[i][k]*c[nc][k]);
	}
    }
  else
    for(i=0;i<grid->Nc;i++) {
      if(grid->ctop[i]<grid->Nk[i])
        coef[i][grid->ctop[i]]=grid->Ac[i]/grid->dzz[i][grid->ctop[i]];
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) 
	coef[i][k] = 2*grid->Ac[i]/(grid->dzz[i][k]+grid->dzz[i][k-1]);
  }
}

/*
 * Function: OperatorQ
 * Usage: OperatorQ(coef,x,y,c,grid,phys,prop);
 * --------------------------------------------
 * Given a vector x, computes the left hand side of the nonhydrostatic pressure
 * Poisson equation and places it into y with y = L(x) for the non-preconditioned
 * solver.
 *
 * The coef array contains coefficients for the vertical derivative terms in the operator.
 * This is computed before the iteration in QCoefficients. The array c stores the preconditioner.
 * The preconditioner stored in c is not used.
 *
 */
static void OperatorQ(REAL **coef, REAL **x, REAL **y, REAL **c, gridT *grid, physT *phys, propT *prop) {

  int i, iptr, k, ne, nf, nc, kmin, kmax;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];
    if(grid->ctop[i]<grid->Nk[i]){
      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	y[i][k]=0;

      for(nf=0;nf<NFACES;nf++) 
	if((nc=grid->neigh[i*NFACES+nf])!=-1) {
	  
	    ne = grid->face[i*NFACES+nf];

	    if(grid->ctop[nc]>grid->ctop[i])
	      kmin = grid->ctop[nc];
	    else
	      kmin = grid->ctop[i];

	    for(k=kmin;k<grid->Nke[ne];k++) 
	      y[i][k]+=(x[nc][k]-x[i][k])*grid->dzz[i][k]*phys->D[ne];
	  }

      for(k=grid->ctop[i]+1;k<grid->Nk[i]-1;k++)
	y[i][k]+=coef[i][k]*x[i][k-1]-(coef[i][k]+coef[i][k+1])*x[i][k]+coef[i][k+1]*x[i][k+1];

      if(grid->ctop[i]<grid->Nk[i]-1) {
	// Top q=0 so q[i][grid->ctop[i]-1]=-q[i][grid->ctop[i]]
	k=grid->ctop[i];
	y[i][k]+=(-2*coef[i][k]-coef[i][k+1])*x[i][k]+coef[i][k+1]*x[i][k+1];

	// Bottom dq/dz = 0 so q[i][grid->Nk[i]]=q[i][grid->Nk[i]-1]
	k=grid->Nk[i]-1;
	y[i][k]+=coef[i][k]*x[i][k-1]-coef[i][k]*x[i][k];
      } else
	y[i][grid->ctop[i]]-=2.0*coef[i][grid->ctop[i]]*x[i][grid->ctop[i]];
    }
  }
}

/*
 * Function: GuessQ
 * Usage: Guessq(q,wold,w,grid,phys,prop,myproc,numprocs,comm);
 * ------------------------------------------------------------
 * Guess a pressure correction field that will enforce the hydrostatic velocity
 * field to speed up the convergence of the pressure Poisson equation.
 *
 */
static void GuessQ(REAL **q, REAL **wold, REAL **w, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm) {
  
  int i, iptr, k;
  REAL qerror;

  // First compute the vertical velocity field that would satisfy continuity
  Continuity(w,grid,phys,prop);

  // Then use this velocity field to back out the required pressure field by
  // integrating w^{n+1}=w*-dt dq/dz and imposing the q=0 boundary condition at
  // the free-surface.
  for(iptr=grid->celldist[0];iptr<grid->celldist[2];iptr++) {
    i = grid->cellp[iptr];
  
    q[i][grid->ctop[i]]=grid->dzz[i][grid->ctop[i]]/2/prop->dt/prop->theta*
      (w[i][grid->ctop[i]]-wold[i][grid->ctop[i]]);
    for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
      q[i][k]=q[i][k-1]+(grid->dzz[i][k]+grid->dzz[i][k-1])/(2.0*prop->dt*prop->theta)*
	(w[i][k]-wold[i][k]);
    }
  }
}

/*
 * Function: Preconditioner
 * Usage: Preconditioner(x,xc,coef,grid,phys,prop);
 * ------------------------------------------------
 * Multiply the vector x by the inverse of the preconditioner M with
 * xc = M^{-1} x
 *
 */
static void Preconditioner(REAL **x, REAL **xc, REAL **coef, gridT *grid, physT *phys, propT *prop) {
  int i, iptr, k, nf, ne, nc, kmin;
  REAL *a = phys->a, *b = phys->b, *c = phys->c, *d = phys->d;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i=grid->cellp[iptr];
    if(grid->ctop[i]<grid->Nk[i]){
      if(grid->ctop[i]<grid->Nk[i]-1) {
        for(k=grid->ctop[i]+1;k<grid->Nk[i]-1;k++) {
	   a[k]=coef[i][k];
	   b[k]=-coef[i][k]-coef[i][k+1];
	   c[k]=coef[i][k+1];
	   d[k]=x[i][k];
        }
      
        // Top q=0 so q[i][grid->ctop[i]-1]=-q[i][grid->ctop[i]]
        k=grid->ctop[i];
        b[k]=-2*coef[i][k]-coef[i][k+1];
        c[k]=coef[i][k+1];
        d[k]=x[i][k];
        
        // Bottom dq/dz = 0 so q[i][grid->Nk[i]]=q[i][grid->Nk[i]-1]
        k=grid->Nk[i]-1;
        a[k]=coef[i][k];
        b[k]=-coef[i][k];
        d[k]=x[i][k];
      
        TriSolve(&(a[grid->ctop[i]]),&(b[grid->ctop[i]]),&(c[grid->ctop[i]]),
        	       &(d[grid->ctop[i]]),&(xc[i][grid->ctop[i]]),grid->Nk[i]-grid->ctop[i]);
       
      } else 
       xc[i][grid->ctop[i]]=-0.5*x[i][grid->ctop[i]]/coef[i][grid->ctop[i]];
    }
  }
}

/*
 * Function: ConditionQ
 * Usage: ConditionQ(x,grid,phys,prop,myproc,comm);
 * ------------------------------------------------
 * Compute the magnitude of the diagonal elements of the coefficient matrix
 * for the pressure-Poisson equation and place it into x after taking its square root.
 *
 */
static void ConditionQ(REAL **x, gridT *grid, physT *phys, propT *prop, int myproc, MPI_Comm comm) {

  int i, iptr, k, ne, nf, nc, kmin, warn=0;
  REAL *a = phys->a;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	x[i][k]=0;

      for(nf=0;nf<NFACES;nf++) 
	if((nc=grid->neigh[i*NFACES+nf])!=-1) {

	  ne = grid->face[i*NFACES+nf];
	  
	  if(grid->ctop[nc]>grid->ctop[i])
	    kmin = grid->ctop[nc];
	  else
	    kmin = grid->ctop[i];
	  
	  for(k=kmin;k<grid->Nke[ne];k++) 
	    x[i][k]+=grid->dzz[i][k]*phys->D[ne];
	}
      
      a[grid->ctop[i]]=grid->Ac[i]/grid->dzz[i][grid->ctop[i]];
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) 
	a[k] = 2*grid->Ac[i]/(grid->dzz[i][k]+grid->dzz[i][k-1]);

      for(k=grid->ctop[i]+1;k<grid->Nk[i]-1;k++)
	x[i][k]+=(a[k]+a[k+1]);

      if(grid->ctop[i]<grid->Nk[i]-1) {
	// Top q=0 so q[i][grid->ctop[i]-1]=-q[i][grid->ctop[i]]
	k=grid->ctop[i];
	x[i][k]+=2*a[k]+a[k+1];

	// Bottom dq/dz = 0 so q[i][grid->Nk[i]]=q[i][grid->Nk[i]-1]
	k=grid->Nk[i]-1;
	x[i][k]+=a[k];
      }
  }

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      for(k=grid->ctop[i];k<grid->Nk[i];k++) {
	if(x[i][k]<=0) {
	  x[i][k]=1;
	  warn=1;
	}
	x[i][k]=sqrt(x[i][k]);
      }
  }
  if(WARNING && warn) printf("Warning...invalid preconditioner!\n");

  // Send the preconditioner to the neighboring processors.
  ISendRecvCellData3D(x,grid,myproc,comm);
}

/*
 * Function: GSSolve
 * Usage: GSSolve(grid,phys,prop,myproc,numprocs,comm);
 * ----------------------------------------------------
 * Solve the free surface equation with a Gauss-Seidell relaxation.
 * This function is used for debugging only.
 *
 */
static void GSSolve(gridT *grid, physT *phys, propT *prop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, iptr, nf, ne, n, niters, *N;
  REAL *h, *hold, *D, *hsrc, myresid, resid, residold, tmp, relax, myNsrc, Nsrc, coef;

  h = phys->h;
  hold = phys->hold;
  D = phys->D;
  hsrc = phys->htmp;
  N = grid->normal;

  tmp = GRAV*pow(prop->theta*prop->dt,2);

  ISendRecvCellData2D(h,grid,myproc,comm);

  relax = prop->relax;
  niters = prop->maxiters;
  resid=0;
  myresid=0;

  myNsrc=0;
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];

    myNsrc+=pow(hsrc[i],2);
  }
  MPI_Reduce(&myNsrc,&(Nsrc),1,MPI_DOUBLE,MPI_SUM,0,comm);
  MPI_Bcast(&Nsrc,1,MPI_DOUBLE,0,comm);
  Nsrc=sqrt(Nsrc);

  for(n=0;n<niters;n++) {

    for(i=0;i<grid->Nc;i++) {
      hold[i] = h[i];
    }

    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      h[i] = hsrc[i];

      coef=1;
      for(nf=0;nf<NFACES;nf++) 
	if(grid->neigh[i*NFACES+nf]!=-1) {
	  ne = grid->face[i*NFACES+nf];

	  coef+=tmp*phys->D[ne]*grid->df[ne]/grid->dg[ne]/grid->Ac[i];
	  h[i]+=relax*tmp*phys->D[ne]*grid->df[ne]/grid->dg[ne]*
	    phys->h[grid->neigh[i*NFACES+nf]]/grid->Ac[i];
	}
      h[i]/=coef;
    }

    residold=resid;
    myresid=0;
    for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
      i = grid->cellp[iptr];

      hold[i] = hsrc[i];

      coef=1;
      for(nf=0;nf<NFACES;nf++) 
	if(grid->neigh[i*NFACES+nf]!=-1) {
	  ne = grid->face[i*NFACES+nf];
	  coef+=tmp*phys->D[ne]*grid->df[ne]/grid->dg[ne]/grid->Ac[i];
	  hold[i]+=tmp*phys->D[ne]*grid->df[ne]/grid->dg[ne]*
	    phys->h[grid->neigh[i*NFACES+nf]]/grid->Ac[i];
	}
      myresid+=pow(hold[i]/coef-h[i],2);
    }
    MPI_Reduce(&myresid,&(resid),1,MPI_DOUBLE,MPI_SUM,0,comm);
    MPI_Bcast(&resid,1,MPI_DOUBLE,0,comm);
    resid=sqrt(resid);

    ISendRecvCellData2D(h,grid,myproc,comm);
    MPI_Barrier(comm);

    if(fabs(resid)<prop->epsilon)
      break;
  }
  if(n==niters && myproc==0 && WARNING) 
    printf("Warning... Iteration not converging after %d steps! RES=%e\n",n,resid);
  
  for(i=0;i<grid->Nc;i++)
    if(h[i]!=h[i]) 
      printf("NaN h[%d] in cgsolve!\n",i);
}

/*
 * Function: UpdateScalars
 * Usage: UpdateScalars(grid,phys,prop,scalar,Cn,kappa,kappaH,kappa_tv,theta);
 * ---------------------------------------------------------------------------
 * Update the scalar quantity stored in the array denoted by scal using the
 * theta method for vertical advection and vertical diffusion and Adams-Bashforth
 * for horizontal advection and diffusion.
 *
 * Cn must store the AB terms from time step n-1 for this scalar
 * kappa denotes the vertical scalar diffusivity
 * kappaH denotes the horizontal scalar diffusivity
 * kappa_tv denotes the vertical turbulent scalar diffusivity
 *
 */
/*
void UpdateScalars(gridT *grid, physT *phys, propT *prop, REAL **scal, REAL **boundary_scal, REAL **Cn, 
		   REAL kappa, REAL kappaH, REAL **kappa_tv, REAL theta,
		   REAL **src1, REAL **src2, REAL *Ftop, REAL *Fbot, int alpha_top, int alpha_bot,
		   MPI_Comm comm, int myproc, int externalforce) 
{
  int i, iptr, j, jptr, ib, k, nf, ktop;
  int Nc=grid->Nc, normal, nc1, nc2, ne;
  REAL df, dg, Ac, dt=prop->dt, fab, *a, *b, *c, *d, *ap, *am, *bd, dznew, mass, *sp, *temp;
  // bing tracer July 2007
  REAL z, Q_solar=1.0e-5, T_scale=5, L_penetration=1, Q_out=0;//7.7e-7;


  ap = phys->ap;
  am = phys->am;
  bd = phys->bp;
  temp = phys->bm;
  a = phys->a;
  b = phys->b;
  c = phys->c;
  d = phys->d;

  if(prop->n==1){ //nov01 || prop->wetdry) {
    fab=1;
    for(i=0;i<grid->Nc;i++)
      for(k=0;k<grid->Nk[i];k++)
	Cn[i][k]=0;
  } else
    fab=1.5;

  for(i=0;i<Nc;i++) 
    for(k=0;k<grid->Nk[i];k++) 
      phys->stmp[i][k]=scal[i][k];

  // Add on boundary fluxes, using stmp2 as the temporary storage
  // variable

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    
    for(k=grid->ctop[i];k<grid->Nk[i];k++)
      phys->stmp2[i][k]=0;
  }

  if(boundary_scal) {
    for(jptr=grid->edgedist[2];jptr<grid->edgedist[5];jptr++) {
      j = grid->edgep[jptr];
      
      ib = grid->grad[2*j];
      
      // Set the value of stmp2 adjacent to the boundary to the value of the boundary.
      // This will be used to add the boundary flux when stmp2 is used again below.
      for(k=grid->ctop[ib];k<grid->Nk[ib];k++)
	phys->stmp2[ib][k]=boundary_scal[jptr-grid->edgedist[2]][k];
    }
  }

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    Ac = grid->Ac[i];
    
    if(grid->ctop[i]>=grid->ctopold[i]) {
      ktop=grid->ctop[i];
      dznew=grid->dzz[i][ktop];
    } else {
      ktop=grid->ctopold[i];
      dznew=0;
      for(k=grid->ctop[i];k<=grid->ctopold[i];k++) 
	dznew+=grid->dzz[i][k];      
    }

    // These are the advective components of the tridiagonal
    // at the new time step.
    for(k=0;k<grid->Nk[i]+1;k++) {
      ap[k] = 0.5*(phys->w[i][k]+fabs(phys->w[i][k]));
      am[k] = 0.5*(phys->w[i][k]-fabs(phys->w[i][k]));
    }
    for(k=ktop+1;k<grid->Nk[i];k++) {
      a[k-ktop]=theta*dt*am[k];
      b[k-ktop]=grid->dzz[i][k]+theta*dt*(ap[k]-am[k+1]);
      c[k-ktop]=-theta*dt*ap[k+1];
    }

    // Top cell advection
    a[0]=0;
    b[0]=dznew-theta*dt*am[ktop+1];
    c[0]=-theta*dt*ap[ktop+1];

    // Bottom cell no-flux boundary condition for advection
    b[(grid->Nk[i]-1)-ktop]+=c[(grid->Nk[i]-1)-ktop];

    // Implicit vertical diffusion terms
    for(k=ktop+1;k<grid->Nk[i];k++)
      bd[k]=(2.0*kappa+kappa_tv[i][k-1]+kappa_tv[i][k])/
	(grid->dzz[i][k-1]+grid->dzz[i][k]);

    for(k=ktop+1;k<grid->Nk[i]-1;k++) {
      a[k-ktop]-=theta*dt*bd[k];
      b[k-ktop]+=theta*dt*(bd[k]+bd[k+1]);
      c[k-ktop]-=theta*dt*bd[k+1];
    }
    if(src1)
      for(k=ktop;k<grid->Nk[i];k++)
	b[k-ktop]+=grid->dzz[i][k]*src1[i][k]*theta*dt;

    // Diffusive fluxes only when more than 1 layer
    if(ktop<grid->Nk[i]-1) {
      // Top cell diffusion
      b[0]+=theta*dt*(bd[ktop+1]+2*alpha_top*bd[ktop+1]);
      c[0]-=theta*dt*bd[ktop+1];

      // Bottom cell diffusion
      a[(grid->Nk[i]-1)-ktop]-=theta*dt*bd[grid->Nk[i]-1];
      b[(grid->Nk[i]-1)-ktop]+=theta*dt*(bd[grid->Nk[i]-1]+2*alpha_bot*bd[grid->Nk[i]-1]);
    }

    // Explicit part into source term d[] 
    for(k=ktop+1;k<grid->Nk[i];k++) 
      d[k-ktop]=grid->dzzold[i][k]*phys->stmp[i][k];
    if(src1)
      for(k=ktop+1;k<grid->Nk[i];k++) 
	d[k-ktop]-=src1[i][k]*(1-theta)*dt*grid->dzzold[i][k]*phys->stmp[i][k];

    d[0]=0;
    if(grid->ctopold[i]<=grid->ctop[i]) {
      for(k=grid->ctopold[i];k<=grid->ctop[i];k++)
	d[0]+=grid->dzzold[i][k]*phys->stmp[i][k];
      if(src1)
	for(k=grid->ctopold[i];k<=grid->ctop[i];k++)
	  d[0]-=src1[i][k]*(1-theta)*dt*grid->dzzold[i][k]*phys->stmp[i][k];
    } else {
      d[0]=grid->dzzold[i][ktop]*phys->stmp[i][ktop];
      if(src1)
	d[0]-=src1[i][ktop]*(1-theta)*dt*grid->dzzold[i][ktop]*phys->stmp[i][k];
    }

    // These are the advective components of the tridiagonal
    // that use the new velocity
    for(k=0;k<grid->Nk[i]+1;k++) {
      ap[k] = 0.5*(phys->wtmp2[i][k]+fabs(phys->wtmp2[i][k]));
      am[k] = 0.5*(phys->wtmp2[i][k]-fabs(phys->wtmp2[i][k]));
    }

    // Explicit advection and diffusion
    for(k=ktop+1;k<grid->Nk[i]-1;k++) 
      d[k-ktop]-=(1-theta)*dt*(am[k]*phys->stmp[i][k-1]+
			       (ap[k]-am[k+1])*phys->stmp[i][k]-
			       ap[k+1]*phys->stmp[i][k+1])+
	(1-theta)*dt*(bd[k]*phys->stmp[i][k-1]
		      -(bd[k]+bd[k+1])*phys->stmp[i][k]
		      +bd[k+1]*phys->stmp[i][k+1]);

    if(ktop<grid->Nk[i]-1) {
      //Flux through bottom of top cell
      k=ktop;
      d[0]=d[0]-(1-theta)*dt*(-am[k+1]*phys->stmp[i][k]-
			   ap[k+1]*phys->stmp[i][k+1])-
	(1-theta)*dt*(-(2*alpha_top*bd[k+1]+bd[k+1])*phys->stmp[i][k]+
		      bd[k+1]*phys->stmp[i][k+1]);
      if(Ftop) d[0]+=dt*(1-alpha_top+2*alpha_top*bd[k+1])*Ftop[i];

      // Through top of bottom cell
      k=grid->Nk[i]-1;
      d[k-ktop]-=(1-theta)*dt*(am[k]*phys->stmp[i][k-1]+
			       ap[k]*phys->stmp[i][k])+
	(1-theta)*dt*(bd[k]*phys->stmp[i][k-1]-
		      (bd[k]+2*alpha_bot*bd[k])*phys->stmp[i][k]);
      if(Fbot) d[k-ktop]+=dt*(-1+alpha_bot+2*alpha_bot*bd[k])*Fbot[i];
    }

    // First add on the source term from the previous time step.
    if(grid->ctop[i]<=grid->ctopold[i]) {
      for(k=grid->ctop[i];k<=grid->ctopold[i];k++) 
	d[0]+=(1-fab)*Cn[i][grid->ctopold[i]]/(1+fabs(grid->ctop[i]-grid->ctopold[i]));
      for(k=grid->ctopold[i]+1;k<grid->Nk[i];k++) 
	d[k-grid->ctopold[i]]+=(1-fab)*Cn[i][k];
    } else {
      for(k=grid->ctopold[i];k<=grid->ctop[i];k++) 
	d[0]+=(1-fab)*Cn[i][k];
      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) 
	d[k-grid->ctop[i]]+=(1-fab)*Cn[i][k];
    }

    for(k=0;k<grid->ctop[i];k++)
      Cn[i][k]=0;

    if(src2)
      for(k=grid->ctop[i];k<grid->Nk[i];k++) 
	Cn[i][k]=dt*src2[i][k]*grid->dzzold[i][k];
    else
      for(k=grid->ctop[i];k<grid->Nk[i];k++)
	Cn[i][k]=0;

    if(externalforce) {
      z=0.0;
      for(k=0;k<grid->ctop[i];k++)
	z-=grid->dz[k];

      k=grid->ctop[i];
      z-=grid->dz[k];
      Cn[i][k-ktop]+=grid->dzz[i][k]*Q_solar/T_scale/L_penetration*exp(0.5*(z-phys->h[i])/L_penetration);//-Q_out*grid->dzz[i][k]/grid->dz[k];

      for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
	z-=0.5*grid->dz[k];
	Cn[i][k-ktop]+=grid->dzz[i][k]*(Q_solar/T_scale*exp(0.5*(z-phys->h[i])/L_penetration));
	z-=0.5*grid->dz[k];
      }
    }


    // Now create the source term for the current time step
    for(k=0;k<grid->Nk[i];k++)
      ap[k]=0;

    for(nf=0;nf<NFACES;nf++) {
      ne = grid->face[i*NFACES+nf];
      normal = grid->normal[i*NFACES+nf];
      df = grid->df[ne];
      dg = grid->dg[ne];
      nc1 = grid->grad[2*ne];
      nc2 = grid->grad[2*ne+1];
      if(nc1==-1) nc1=nc2;
      if(nc2==-1) {
	nc2=nc1;
	if(boundary_scal && grid->mark[ne]==2)
	  sp=phys->stmp2[nc1];
	else
	  sp=phys->stmp[nc1];
      } else 
	sp=phys->stmp[nc2];

      for(k=0;k<grid->Nke[ne];k++) 
	temp[k]=UpWind(phys->utmp2[ne][k],
		       grid->dzzold[nc1][k]*phys->stmp[nc1][k],
		       grid->dzzold[nc2][k]*sp[k]);

      //if(prop->wetdry) {   //bing nov01
	for(k=0;k<grid->Nke[ne];k++)
	  ap[k] += dt*df*normal/Ac*
	    (theta*phys->u[ne][k]+(1-theta)*phys->utmp2[ne][k])*temp[k];
	} 
        else {
	for(k=0;k<grid->Nk[nc2];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]+fabs(phys->utmp2[ne][k]))*
	    sp[k]*grid->dzzold[nc2][k];
	for(k=0;k<grid->Nk[nc1];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]-fabs(phys->utmp2[ne][k]))*
	    phys->stmp[nc1][k]*grid->dzzold[nc1][k];
	    }
}

    for(k=ktop+1;k<grid->Nk[i];k++) 
      Cn[i][k-ktop]-=ap[k];
    
    for(k=0;k<=ktop;k++) 
      Cn[i][0]-=ap[k];

    // Add on the source from the current time step to the rhs.
    for(k=0;k<grid->Nk[i]-ktop;k++) 
      d[k]+=fab*Cn[i][k];

    for(k=ktop;k<grid->Nk[i];k++)
      ap[k]=Cn[i][k-ktop];
    for(k=0;k<=ktop;k++)
      Cn[i][k]=0;
    for(k=ktop+1;k<grid->Nk[i];k++)
      Cn[i][k]=ap[k];
    for(k=grid->ctop[i];k<=ktop;k++)
      Cn[i][k]=ap[ktop]/(1+fabs(grid->ctop[i]-ktop));

    if(grid->Nk[i]-ktop>1) 
      TriSolve(a,b,c,d,&(scal[i][ktop]),grid->Nk[i]-ktop);
    else if(b[0]!=0)
      scal[i][ktop]=d[0]/b[0];

    for(k=0;k<grid->ctop[i];k++)
      scal[i][k]=0;

    for(k=grid->ctop[i];k<grid->ctopold[i];k++) 
      scal[i][k]=scal[i][ktop];
  }
}
*/

/*
 * Function: Continuity
 * Usage: Continuity(w,grid,phys,prop);
 * ------------------------------------
 * Compute the vertical velocity field that satisfies continuity.  Use
 * the upwind flux face heights to ensure consistency with continuity.
 *
 */

static void Continuity(REAL **w, gridT *grid, physT *phys, propT *prop)
{
  int i, k, nf, iptr, ne, nc1, nc2, j, jptr;
  REAL ap, am, d;
  //printf("Inside continuity \n"); 
  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];

    if(grid->ctop[i]<grid->Nk[i]){
      for(k=0;k<grid->ctop[i];k++) 
        w[i][k] = 0;
  
      w[i][grid->Nk[i]] = 0;
      d=grid->dzz[i][grid->Nk[i]-1];
      for(k=grid->Nk[i]-1;k>=grid->ctop[i];k--) {
        w[i][k] = w[i][k+1];
        for(nf=0;nf<NFACES;nf++) {
  	ne = grid->face[i*NFACES+nf];
  	nc1 = grid->grad[2*ne];
  	nc2 = grid->grad[2*ne+1];
  	if(nc1==-1) nc1=nc2;
  	if(nc2==-1) nc2=nc1;
  
  	ap=0;
  	if(k<grid->Nk[nc2])
  	  ap=0.5*(phys->u[ne][k]+fabs(phys->u[ne][k]))*grid->dzz[nc2][k];
  
  	am=0;
  	if(k<grid->Nk[nc1])
  	  am=0.5*(phys->u[ne][k]-fabs(phys->u[ne][k]))*grid->dzz[nc1][k];
  
  	w[i][k]-=(ap+am)*grid->df[ne]*grid->normal[i*NFACES+nf]/grid->Ac[i];
        }
      }
    }
  }
}

/*
 * Function: ComputeConservatives
 * Usage: ComputeConservatives(grid,phys,prop,myproc,numprocs,comm);
 * -----------------------------------------------------------------
 * Compute the total mass, volume, and potential energy within the entire
 * domain and return a warning if the mass and volume are not conserved to within
 * the tolerance CONSERVED specified in suntans.h 
 *
 */
static void ComputeConservatives(gridT *grid, physT *phys, propT *prop, int myproc, int numprocs,
			  MPI_Comm comm)
{
  int i, iptr, k;
  REAL mass, volume, volh, height, Ep;

  if(myproc==0) phys->mass=0;
  if(myproc==0) phys->volume=0;
  if(myproc==0) phys->Ep=0;

  // volh is the horizontal integral of h+d, whereas vol is the
  // 3-d integral of dzz
  mass=0;
  volume=0;
  volh=0;
  Ep=0;

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++) {
    i = grid->cellp[iptr];
    height = 0;
    volh+=grid->Ac[i]*(grid->dv[i]+phys->h[i]);
    Ep+=0.5*GRAV*grid->Ac[i]*(phys->h[i]+grid->dv[i])*(phys->h[i]-grid->dv[i]);
    for(k=grid->ctop[i];k<grid->Nk[i];k++) {
      height += grid->dzz[i][k];
      volume+=grid->Ac[i]*grid->dzz[i][k];
      mass+=phys->s[i][k]*grid->Ac[i]*grid->dzz[i][k];
    }
  }

  // Comment out the volh reduce if that integral is desired.  The
  // volume integral is used since the volh integral is useful
  // only for debugging.
  MPI_Reduce(&mass,&(phys->mass),1,MPI_DOUBLE,MPI_SUM,0,comm);
  //MPI_Reduce(&volh,&(phys->volume),1,MPI_DOUBLE,MPI_SUM,0,comm);
  MPI_Reduce(&volume,&(phys->volume),1,MPI_DOUBLE,MPI_SUM,0,comm);
  MPI_Reduce(&Ep,&(phys->Ep),1,MPI_DOUBLE,MPI_SUM,0,comm);

  // Compare the quantities to the original values at the beginning of the
  // computation.  If prop->n==0 (beginning of simulation), then store the
  // starting values for comparison.
  if(myproc==0) {
    if(prop->n==0) {
      phys->volume0 = phys->volume;
      phys->mass0 = phys->mass;
      phys->Ep0 = phys->Ep;
    } else {
      if(fabs((phys->volume-phys->volume0)/phys->volume0)>CONSERVED && prop->volcheck)
	printf("Warning! Not volume conservative! V(0)=%e, V(t)=%e\n",
	       phys->volume0,phys->volume);
      if(fabs((phys->mass-phys->mass0)/phys->volume0)>CONSERVED && prop->masscheck) 
	printf("Warning! Not mass conservative! M(0)=%e, M(t)=%e\n",
	       phys->mass0,phys->mass);
    }
  }
}

/*
 * Function: Check
 * Usage: Check(grid,phys,prop,myproc,numprocs,comm);
 * --------------------------------------------------
 * Check to make sure the run isn't blowing up.
 *
 */    
static int Check(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, 
		 waveT *wave, wpropT *wprop, int myproc, int numprocs, MPI_Comm comm)
{
  int i, k, icu, kcu, icw, kcw, Nc=grid->Nc, Ne=grid->Ne, ih, is, ks, iu, ku;
  int uflag=1, sflag=1, hflag=1, myalldone, alldone;
  REAL C, CmaxU, CmaxW;

  icu=kcu=icw=kcw=ih=is=ks=iu=ku=0;

  for(i=0;i<Nc;i++) 
    if(phys->h[i]!=phys->h[i]) {
	hflag=0;
	ih=i;
	break;
    }

  /*for(i=0;i<Nc;i++) {
    for(k=0;k<grid->Nk[i];k++)
      if(phys->s[i][k]!=phys->s[i][k]) {
	sflag=0;
	is=i;
	ks=k;
	break;
      }
    if(!sflag)
      break;
  }*/

  for(i=0;i<Ne;i++) {
    for(k=0;k<grid->Nke[i];k++)
      if(phys->u[i][k]!=phys->u[i][k]) {
	uflag=0;
	iu=i;
	ku=k;
	break;
      }
    if(!uflag)
      break;
  }

  CmaxU=0;
  for(i=0;i<Ne;i++) 
    for(k=grid->etop[i]+1;k<grid->Nke[i];k++) {
      C = fabs(phys->u[i][k])*prop->dt/grid->dg[i];
      if(C>CmaxU) {
	icu = i;
	kcu = k;
	CmaxU = C;
      }
    }

  CmaxW=0;
  for(i=0;i<Nc;i++) 
    for(k=grid->ctop[i]+1;k<grid->Nk[i];k++) {
      if(grid->dzz[i][k]>0){
        C = 0.5*fabs(phys->w[i][k]+phys->w[i][k+1])*prop->dt/grid->dzz[i][k];
        if(C>CmaxW) {
	  icw = i;
	  kcw = k;
	  CmaxW = C;
        }
      }
    }
  
  myalldone=0;
  if(!uflag || !sflag || !hflag || CmaxU>prop->Cmax || CmaxW>prop->Cmax) {
    printf("Time step %d: Processor %d, Run is blowing up!\n",prop->n,myproc);
    
    if(CmaxU>prop->Cmax)
      printf("Courant number problems at (%d,%d), Umax=%f, dx=%f CmaxU=%.2f > %.2f\n",
	     icu,kcu,phys->u[icu][kcu],grid->dg[icu],CmaxU,prop->Cmax);
    else if(CmaxW>prop->Cmax)
      printf("Courant number problems at (%d,%d), Wmax=%f, dz=%f CmaxW=%.2f > %.2f\n",
	     icw,kcw,0.5*(phys->w[icw][kcw]+phys->w[icw][kcw+1]),grid->dzz[icw][kcw],CmaxW,prop->Cmax);
    else
      printf("Courant number is okay: CmaxU=%.2f,CmaxW=%.2f < %.2f\n",CmaxU,CmaxW,prop->Cmax);

    if(!uflag)
      printf("U is divergent at (%d,%d)\n",iu,ku);
    else
      printf("U is okay.\n");
    if(!sflag) 
      printf("Scalar is divergent at (%d,%d).\n",is,ks);
    else
      printf("Scalar is okay.\n");
    if(!hflag)
      printf("Free-surface is divergent at (%d)\n",ih);
    else
      printf("Free-surface is okay.\n");
    
    OutputData(grid,phys,prop,sedi,sprop, wave,wprop, myproc,numprocs,1,comm);
    myalldone=1;
  }

  MPI_Reduce(&myalldone,&alldone,1,MPI_INT,MPI_SUM,0,comm);
  MPI_Bcast(&alldone,1,MPI_INT,0,comm);

  return alldone;
}

/* 
 * Function: ComputeVelocityVector
 * Usage: ComputeVelocityVector(u,uc,vc,grid);
 * -------------------------------------------
 * Compute the cell-centered components of the velocity vector and place them
 * into uc and vc.  This function estimates the velocity vector with
 *
 * u = 1/Area * Sum_{faces} u_{face} normal_{face} df_{face}/d_{ef,face}
 *
 */

static void ComputeVelocityVector(physT *phys, REAL **uc, REAL **vc, gridT *grid) {
 
  int k, n, ne1,ne2, nf, ne;
  REAL sum;
  REAL un[3],det[3],ux[3],uy[3],vtx[3],vty[3],al[3],ptx[3],pty[3],xs[2],us[2];
  int jp[]={1,2,0}, jm[]={2,0,1};
  
  for(n=0;n<grid->Nc;n++) {
    for(k=0;k<grid->Nk[n];k++) {
	xs[0]=grid->xv[n];	
	xs[1]=grid->yv[n];
	us[0]=us[1]=0;
	StableInterpVelo(grid,phys, n, k,xs,us);	
	uc[n][k]=us[0];
	vc[n][k]=us[1];
      }
  }
}

/*
 * Function: OutputData
 * Usage: OutputData(grid,phys,prop,myproc,numprocs,blowup,comm);
 * --------------------------------------------------------------
 * Output the data every ntout steps as specified in suntans.dat
 * If this is the last time step or if the run is blowing up (blowup==1),
 * then output the data to the restart file specified by the file pointer
 * prop->StoreFID.
 *
 * If ASCII is specified then the data is output in ascii format, otherwise
 * it is output in binary format.  This variable is specified in suntans.h.
 *
 */
static void OutputData(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave, wpropT *wprop,
		int myproc, int numprocs, int blowup, MPI_Comm comm)
{
  int i, j, k, index, nwritten;
  int m, n, l, s; 
  REAL z;
  REAL *tmp = (REAL *)SunMalloc(grid->Ne*sizeof(REAL),"OutputData");

  // change prop->n==prop->nstart+1 to prop->nstart to output initial condition 
  if(!(prop->n%prop->ntout) || prop->n==prop->nstart || blowup|| prop->n==prop->nsteps+prop->nstart) {
                                 //^tsungwei
    if(myproc==0 && VERBOSE>=1) 
      if(!blowup) 
	printf("Outputting data at step %d of %d\n",prop->n,prop->nsteps+prop->nstart);
      else
	printf("Outputting blowup data at step %d of %d\n",prop->n,prop->nsteps+prop->nstart);


    if(ASCII) 
      for(i=0;i<grid->Nc;i++)
	fprintf(prop->FreeSurfaceFID,"%f\n",phys->h[i]);
    else {
      nwritten=fwrite(phys->h,sizeof(REAL),grid->Nc,prop->FreeSurfaceFID);
      if(nwritten!=grid->Nc) {
	printf("Error outputting free-surface data!\n");
	exit(EXIT_WRITING);
      }
    }
    fflush(prop->FreeSurfaceFID);
    
    if (prop->sedi){
      if(ASCII) 
	for(i=0;i<grid->Nc;i++)
	  fprintf(prop->SediDepositionFID,"%f\n",sedi->dpsit[i]);
      else {
	nwritten=fwrite(sedi->dpsit,sizeof(REAL),grid->Nc,prop->SediDepositionFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting sediment deposition data!\n");
	  exit(EXIT_WRITING);
	}
      }
      fflush(prop->SediDepositionFID);

      if(ASCII) 
	for(i=0;i<grid->Nc;i++) {
	  for(k=0;k<grid->Nk[i];k++)       
	    fprintf(prop->SSCFID,"%e\n",sedi->sdtot[i][k]);
	  for(k=grid->Nk[i];k<grid->Nkmax;k++)
	    fprintf(prop->SSCFID,"0.0\n");
	}
      else 
	for(k=0;k<grid->Nkmax;k++) {
	  for(i=0;i<grid->Nc;i++) {
	    if(k<grid->Nk[i])
	      phys->htmp[i]=sedi->sdtot[i][k];
	    else
	      phys->htmp[i]=EMPTY;
	  }
	  nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->SSCFID);
	  if(nwritten!=grid->Nc) {
	    printf("Error outputting SSC data!\n");
	    exit(EXIT_WRITING);
	  }
	}
      fflush(prop->SSCFID);
//////Kurt Nelson Added
//For class zero
      if(ASCII)
        for(i=0;i<grid->Nc;i++) {
          for(k=0;k<grid->Nk[i];k++)
            fprintf(prop->SSCZeroFID,"%e\n",sedi->scc0[i][k]);
          for(k=grid->Nk[i];k<grid->Nkmax;k++)
            fprintf(prop->SSCZeroFID,"0.0\n");
        }
      else
        for(k=0;k<grid->Nkmax;k++) {
          for(i=0;i<grid->Nc;i++) {
            if(k<grid->Nk[i])
              phys->htmp[i]=sedi->scc0[i][k];
            else
              phys->htmp[i]=EMPTY;
          }
          nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->SSCZeroFID);
          if(nwritten!=grid->Nc) {
            printf("Error outputting SSCZero data!\n");
            exit(EXIT_WRITING);
          }
        }
      fflush(prop->SSCZeroFID);

//For Class one
      if(ASCII)
        for(i=0;i<grid->Nc;i++) {
          for(k=0;k<grid->Nk[i];k++)
            fprintf(prop->SSCOneFID,"%e\n",sedi->scc1[i][k]);
          for(k=grid->Nk[i];k<grid->Nkmax;k++)
            fprintf(prop->SSCOneFID,"0.0\n");
        }
      else
        for(k=0;k<grid->Nkmax;k++) {
          for(i=0;i<grid->Nc;i++) {
            if(k<grid->Nk[i])
              phys->htmp[i]=sedi->scc1[i][k];
            else
              phys->htmp[i]=EMPTY;
          }
          nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->SSCOneFID);
          if(nwritten!=grid->Nc) {
            printf("Error outputting SSCOne data!\n");
            exit(EXIT_WRITING);
          }
        }
      fflush(prop->SSCOneFID);
///////////////////////////////////////////////////
    }
    if (prop->wave){
      if(ASCII) 
	for(i=0;i<grid->Nc;i++)
	  fprintf(prop->WaveHeightFID,"%f\n",wave->Hs[i]);
      else {
	nwritten=fwrite(wave->Hs, sizeof(REAL),grid->Nc,prop->WaveHeightFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting wave height data!\n");
	  exit(EXIT_WRITING);
	}
      }
      fflush(prop->WaveHeightFID);
   
     
      if(ASCII) 
	for(i=0;i<grid->Nc;i++)
	  fprintf(prop->WindSpeedFID,"%f\n",wave->wind_spf[i]);
      else {
	nwritten=fwrite(wave->wind_spf, sizeof(REAL),grid->Nc,prop->WindSpeedFID);	
	if(nwritten!=grid->Nc) {
	  printf("Error outputting wind speed data!\n");
	  exit(EXIT_WRITING);
	}
      }
      fflush(prop->WindSpeedFID);
   
      if(ASCII) 
	for(i=0;i<grid->Nc;i++)
	  fprintf(prop->WindDirectionFID,"%f\n",wave->wind_dgf[i]);

      else {
	nwritten=fwrite(wave->wind_dgf, sizeof(REAL),grid->Nc,prop->WindDirectionFID);

	if(nwritten!=grid->Nc) {
	  printf("Error outputting wind direction data!\n");
	  exit(EXIT_WRITING);
	}
      }
      fflush(prop->WindDirectionFID);
    
      if(ASCII) 
	for(i=0;i<grid->Nc;i++)
	  fprintf(prop->WaveVelocityFID,"%f\n",wave->thtamean[i]);
      else {
	nwritten=fwrite(wave->thtamean, sizeof(REAL),grid->Nc,prop->WaveVelocityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting wave velocity data!\n");
	  exit(EXIT_WRITING);
	}
	fflush(prop->WaveVelocityFID);
      }
    }
    // ut stores the tangential component of velocity on the faces.
    if(ASCII) 
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i];k++) 
	  fprintf(prop->HorizontalVelocityFID,"%e %e %e\n",
		  phys->uc[i][k],phys->vc[i][k],0.5*(phys->w[i][k]+phys->w[i][k+1]));
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->HorizontalVelocityFID,"0.0 0.0 0.0\n");
      }
    else 
      for(k=0;k<grid->Nkmax;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i]) 
	    tmp[i]=phys->uc[i][k];
	  else
	    tmp[i]=0;
	}
	nwritten=fwrite(tmp,sizeof(REAL),grid->Nc,prop->HorizontalVelocityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting Horizontal Velocity data!\n");
	  exit(EXIT_WRITING);
	}
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    tmp[i]=phys->vc[i][k];
	  else
	    tmp[i]=0;
	}
	nwritten=fwrite(tmp,sizeof(REAL),grid->Nc,prop->HorizontalVelocityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting Horizontal Velocity data!\n");
	  exit(EXIT_WRITING);
	}
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    tmp[i]=0.5*(phys->w[i][k]+phys->w[i][k+1]);
	  else
	    tmp[i]=0;
	}
	nwritten=fwrite(tmp,sizeof(REAL),grid->Nc,prop->HorizontalVelocityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting Face Velocity data!\n");
	  exit(EXIT_WRITING);
	}
      }
    fflush(prop->HorizontalVelocityFID);
        
    /*
    if(ASCII)
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i]+1;k++)
	  fprintf(prop->VerticalVelocityFID,"%e\n",phys->w[i][k]);
	for(k=grid->Nk[i]+1;k<grid->Nkmax+1;k++)
	  fprintf(prop->VerticalVelocityFID,"0.0\n");
      }
    else {
      for(k=0;k<grid->Nkmax+1;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i]+1)
	    phys->htmp[i]=phys->w[i][k];
	  else
	    phys->htmp[i]=EMPTY;
	}
	nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->VerticalVelocityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting vertical velocity data!\n");
	  exit(EXIT_WRITING);
	}
      }
    }
    fflush(prop->VerticalVelocityFID);
    */

    if(ASCII) {
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i];k++)
	  fprintf(prop->SalinityFID,"%e\n",phys->s[i][k]);
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->SalinityFID,"0.0\n");
      }
      if(prop->n==1+prop->nstart) {
	for(i=0;i<grid->Nc;i++) {
	  for(k=0;k<grid->Nk[i];k++)
	    fprintf(prop->BGSalinityFID,"%e\n",phys->s0[i][k]);
	  for(k=grid->Nk[i];k<grid->Nkmax;k++)
	    fprintf(prop->BGSalinityFID,"0.0\n");
	}
      }
    } else {
      for(k=0;k<grid->Nkmax;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    phys->htmp[i]=phys->s[i][k];
	  else
	    phys->htmp[i]=EMPTY;
	}
	nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->SalinityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting salinity data!\n");
	  exit(EXIT_WRITING);
	}
      }
      if(prop->n==1+prop->nstart) {
	for(k=0;k<grid->Nkmax;k++) {
	  for(i=0;i<grid->Nc;i++) {
	    if(k<grid->Nk[i])
	      phys->htmp[i]=phys->s0[i][k];
	    else
	      phys->htmp[i]=EMPTY;
	  }
	  nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->BGSalinityFID);
	  if(nwritten!=grid->Nc) {
	    printf("Error outputting background salinity data!\n");
	    exit(EXIT_WRITING);
	  }
	}
      }
    }
    fflush(prop->SalinityFID);
 
    if(ASCII) 
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i];k++)
	  fprintf(prop->TemperatureFID,"%e\n",phys->T[i][k]);	  
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->TemperatureFID,"0.0\n");
      }
    else 
      for(k=0;k<grid->Nkmax;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    phys->htmp[i]=phys->T[i][k];
	  else
	    phys->htmp[i]=EMPTY;
	}
	nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->TemperatureFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting temperature data!\n");
	  exit(EXIT_WRITING);
	}
      }
    fflush(prop->TemperatureFID);
   
    /*
    if(ASCII) 
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i];k++)
	  fprintf(prop->PressureFID,"%e\n",phys->q[i][k]);
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->PressureFID,"0.0\n");
      }
    else 
      for(k=0;k<grid->Nkmax;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    phys->htmp[i]=phys->q[i][k];
	  else
	    phys->htmp[i]=EMPTY;
	}
	nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->PressureFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting pressure data!\n");
	  exit(EXIT_WRITING);
	}
      }
    fflush(prop->PressureFID);
    */
    
    if(prop->turbmodel) {
      if(ASCII) 
	for(i=0;i<grid->Nc;i++) {
	  for(k=0;k<grid->Nk[i];k++)
	    fprintf(prop->EddyViscosityFID,"%e\n",phys->nu_tv[i][k]);
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->EddyViscosityFID,"0.0\n");
	}
      else 
	for(k=0;k<grid->Nkmax;k++) {
	  for(i=0;i<grid->Nc;i++) {
	    if(k<grid->Nk[i])
	      phys->htmp[i]=phys->nu_tv[i][k];
	    else
	      phys->htmp[i]=EMPTY;
	  }
	  nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->EddyViscosityFID);
	  if(nwritten!=grid->Nc) {
	    printf("Error outputting eddy viscosity data!\n");
	    exit(EXIT_WRITING);
	  }
	}
      fflush(prop->EddyViscosityFID);
      //    }
   
     
    if(ASCII) 
      for(i=0;i<grid->Nc;i++) {
	for(k=0;k<grid->Nk[i];k++)
	  fprintf(prop->ScalarDiffusivityFID,"%e\n",phys->kappa_tv[i][k]);
	for(k=grid->Nk[i];k<grid->Nkmax;k++)
	  fprintf(prop->ScalarDiffusivityFID,"0.0\n");
      }
    else 
      for(k=0;k<grid->Nkmax;k++) {
	for(i=0;i<grid->Nc;i++) {
	  if(k<grid->Nk[i])
	    phys->htmp[i]=phys->kappa_tv[i][k];
	  else
	    phys->htmp[i]=EMPTY;
	}
	nwritten=fwrite(phys->htmp,sizeof(REAL),grid->Nc,prop->ScalarDiffusivityFID);
	if(nwritten!=grid->Nc) {
	  printf("Error outputting scalar diffusivity data!\n");
	  exit(EXIT_WRITING);
	}
      }
    fflush(prop->ScalarDiffusivityFID);
    }
  /*
    for(i=0;i<grid->Nc;i++) {
      for(k=0;k<grid->Nk[i];k++)
	fprintf(prop->VerticalGridFID,"%e\n",grid->dzz[i][k]);
      for(k=grid->Nk[i];k<grid->Nkmax;k++)
	fprintf(prop->VerticalGridFID,"0.0\n");
	}*/
  }
  /*
  fflush(prop->VerticalGridFID);
  */
  if(prop->n==prop->nstart+1)
    fclose(prop->BGSalinityFID);

  //if(prop->n==prop->nsteps+prop->nstart || blowup) {
  if(!(prop->n%(prop->ntout)) || prop->n==prop->nsteps+prop->nstart || blowup){  
 
    if(VERBOSE>=1 && myproc==0) printf("Writing to rstore...\n");
    fseek( prop->StoreFID, 0, SEEK_SET );
    nwritten=fwrite(&(prop->n),sizeof(int),1,prop->StoreFID);
    
    fwrite(phys->h,sizeof(REAL),grid->Nc,prop->StoreFID);
    for(j=0;j<grid->Ne;j++) 
      fwrite(phys->Cn_U[j],sizeof(REAL),grid->Nke[j],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->Cn_W[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->Cn_R[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->Cn_T[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);

    if(prop->turbmodel) {
      for(i=0;i<grid->Nc;i++) 
	fwrite(phys->Cn_q[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
      for(i=0;i<grid->Nc;i++) 
	fwrite(phys->Cn_l[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);

      for(i=0;i<grid->Nc;i++) 
	fwrite(phys->qT[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
      for(i=0;i<grid->Nc;i++) 
	fwrite(phys->lT[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    }
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->nu_tv[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->kappa_tv[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);

    for(j=0;j<grid->Ne;j++) 
      fwrite(phys->u[j],sizeof(REAL),grid->Nke[j],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->w[i],sizeof(REAL),grid->Nk[i]+1,prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->q[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);

    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->s[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->T[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    for(i=0;i<grid->Nc;i++) 
      fwrite(phys->s0[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
    //Store wave and sediment variables. By YJ

    if (prop->wave)
      for(m=0; m<wprop->Mw; m++)
	for(n=0; n<wprop->Nw; n++)
	  fwrite(wave->N[m][n],sizeof(REAL),grid->Nc,prop->StoreFID);

    if (prop->sedi){
      for(s=0; s<sprop->Nsize; s++)
	for(i=0; i<grid->Nc; i++)
	  fwrite(sedi->sd[s][i],sizeof(REAL),grid->Nk[i],prop->StoreFID);

      for(i=0; i<grid->Nc; i++){
	fwrite(sedi->sdtot[i],sizeof(REAL),grid->Nk[i],prop->StoreFID);
	fwrite(sedi->M[i],sizeof(REAL),sprop->NL,prop->StoreFID);
      }
      
    }
    fflush(prop->StoreFID);

  }
  
  if(prop->n==prop->nsteps+prop->nstart) {
    fclose(prop->FreeSurfaceFID);
    fclose(prop->WindSpeedFID);
    fclose(prop->WindDirectionFID);
    fclose(prop->HorizontalVelocityFID);
    fclose(prop->PressureFID);
    fclose(prop->VerticalVelocityFID);
    fclose(prop->SalinityFID);
    //fclose(prop->VerticalGridFID);
    fclose(prop->StoreFID);
  }



  SunFree(tmp,grid->Ne*sizeof(REAL),"OutputData");

}

/*
 * Function: ReadProperties
 * Usage: ReadProperties(prop,myproc);
 * -----------------------------------
 * This function reads in the properties specified in the suntans.dat
 * data file.
 *
 */
void ReadProperties(propT **prop, int myproc)
{
  *prop = (propT *)SunMalloc(sizeof(propT),"ReadProperties");
  
  (*prop)->thetaramptime = MPI_GetValue(DATAFILE,"thetaramptime","ReadProperties",myproc);
  (*prop)->theta = MPI_GetValue(DATAFILE,"theta","ReadProperties",myproc);
  (*prop)->thetaS = MPI_GetValue(DATAFILE,"thetaS","ReadProperties",myproc);
  (*prop)->thetaB = MPI_GetValue(DATAFILE,"thetaB","ReadProperties",myproc);
  (*prop)->beta = MPI_GetValue(DATAFILE,"beta","ReadProperties",myproc);
  (*prop)->kappa_s = MPI_GetValue(DATAFILE,"kappa_s","ReadProperties",myproc);
  (*prop)->kappa_sH = MPI_GetValue(DATAFILE,"kappa_sH","ReadProperties",myproc);
  (*prop)->gamma = MPI_GetValue(DATAFILE,"gamma","ReadProperties",myproc);
  (*prop)->kappa_T = MPI_GetValue(DATAFILE,"kappa_T","ReadProperties",myproc);
  (*prop)->kappa_TH = MPI_GetValue(DATAFILE,"kappa_TH","ReadProperties",myproc);
  (*prop)->nu = MPI_GetValue(DATAFILE,"nu","ReadProperties",myproc);
  (*prop)->nu_H = MPI_GetValue(DATAFILE,"nu_H","ReadProperties",myproc);
  (*prop)->tau_Tp = MPI_GetValue(DATAFILE,"tau_T","ReadProperties",myproc);
  (*prop)->z0T = MPI_GetValue(DATAFILE,"z0T","ReadProperties",myproc);
  (*prop)->z0B = MPI_GetValue(DATAFILE,"z0B","ReadProperties",myproc);
  (*prop)->CdT = MPI_GetValue(DATAFILE,"CdT","ReadProperties",myproc);
  (*prop)->CdB = MPI_GetValue(DATAFILE,"CdB","ReadProperties",myproc);
  (*prop)->CdW = MPI_GetValue(DATAFILE,"CdW","ReadProperties",myproc);
  (*prop)->turbmodel = (int)MPI_GetValue(DATAFILE,"turbmodel","ReadProperties",myproc);
  (*prop)->dt = MPI_GetValue(DATAFILE,"dt","ReadProperties",myproc);
  (*prop)->Cmax = MPI_GetValue(DATAFILE,"Cmax","ReadProperties",myproc);
  (*prop)->nsteps = (int)MPI_GetValue(DATAFILE,"nsteps","ReadProperties",myproc);
  (*prop)->ntout = (int)MPI_GetValue(DATAFILE,"ntout","ReadProperties",myproc);
  (*prop)->ntprog = (int)MPI_GetValue(DATAFILE,"ntprog","ReadProperties",myproc);
  (*prop)->ntconserve = (int)MPI_GetValue(DATAFILE,"ntconserve","ReadProperties",myproc);
  (*prop)->nonhydrostatic = (int)MPI_GetValue(DATAFILE,"nonhydrostatic","ReadProperties",myproc);
  (*prop)->cgsolver = (int)MPI_GetValue(DATAFILE,"cgsolver","ReadProperties",myproc);
  (*prop)->maxiters = (int)MPI_GetValue(DATAFILE,"maxiters","ReadProperties",myproc);
  (*prop)->qmaxiters = (int)MPI_GetValue(DATAFILE,"qmaxiters","ReadProperties",myproc);
  (*prop)->qprecond = (int)MPI_GetValue(DATAFILE,"qprecond","ReadProperties",myproc);
  (*prop)->epsilon = MPI_GetValue(DATAFILE,"epsilon","ReadProperties",myproc);
  (*prop)->qepsilon = MPI_GetValue(DATAFILE,"qepsilon","ReadProperties",myproc);
  (*prop)->resnorm = MPI_GetValue(DATAFILE,"resnorm","ReadProperties",myproc);
  (*prop)->relax = MPI_GetValue(DATAFILE,"relax","ReadProperties",myproc);
  (*prop)->amp = MPI_GetValue(DATAFILE,"amp","ReadProperties",myproc);
  (*prop)->omega = MPI_GetValue(DATAFILE,"omega","ReadProperties",myproc);
  (*prop)->timescale = MPI_GetValue(DATAFILE,"timescale","ReadProperties",myproc);
  (*prop)->flux = MPI_GetValue(DATAFILE,"flux","ReadProperties",myproc);
  (*prop)->volcheck = MPI_GetValue(DATAFILE,"volcheck","ReadProperties",myproc);
  (*prop)->masscheck = MPI_GetValue(DATAFILE,"masscheck","ReadProperties",myproc);
  (*prop)->nonlinear = MPI_GetValue(DATAFILE,"nonlinear","ReadProperties",myproc);
  (*prop)->newcells = MPI_GetValue(DATAFILE,"newcells","ReadProperties",myproc);
  (*prop)->wetdry = MPI_GetValue(DATAFILE,"wetdry","ReadProperties",myproc);
  (*prop)->Coriolis_f = MPI_GetValue(DATAFILE,"Coriolis_f","ReadProperties",myproc);
  (*prop)->sponge_distance = MPI_GetValue(DATAFILE,"sponge_distance","ReadProperties",myproc);
  (*prop)->sponge_decay = MPI_GetValue(DATAFILE,"sponge_decay","ReadProperties",myproc);
  (*prop)->readSalinity = MPI_GetValue(DATAFILE,"readSalinity","ReadProperties",myproc);
  (*prop)->readTemperature = MPI_GetValue(DATAFILE,"readTemperature","ReadProperties",myproc);
  (*prop)->dt_tideBC = MPI_GetValue(DATAFILE,"dt_tideBC","ReadProperties",myproc);
  (*prop)->wave = (int)MPI_GetValue(DATAFILE,"wave","ReadProperties",myproc);
  (*prop)->sedi = (int)MPI_GetValue(DATAFILE,"sedi","ReadProperties",myproc);

}

/* 
 * Function: OpenFiles
 * Usage: OpenFiles(prop,myproc);
 * ------------------------------
 * Open all of the files used for i/o to store the file pointers.
 *
 */
void OpenFiles(propT *prop, int myproc)
{
  char str[BUFFERLENGTH], filename[BUFFERLENGTH];

  if(prop->readSalinity) {
    MPI_GetFile(filename,DATAFILE,"InitSalinityFile","OpenFiles",myproc);
    prop->InitSalinityFID = MPI_FOpen(filename,"r","OpenFiles",myproc);
  }
  if(prop->readTemperature) {
    MPI_GetFile(filename,DATAFILE,"InitTemperatureFile","OpenFiles",myproc);
    prop->InitTemperatureFID = MPI_FOpen(filename,"r","OpenFiles",myproc);
  }

  MPI_GetFile(filename,DATAFILE,"FreeSurfaceFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->FreeSurfaceFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"SediDepositionFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->SediDepositionFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"SSCFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->SSCFID = MPI_FOpen(str,"w","OpenFiles",myproc);
/////////////////////////Kurt Nelson Added

  MPI_GetFile(filename,DATAFILE,"SSCZeroFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->SSCZeroFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"SSCOneFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->SSCOneFID = MPI_FOpen(str,"w","OpenFiles",myproc);

//////////////////////////////////////////////////////////////////////////
  MPI_GetFile(filename,DATAFILE,"WaveHeightFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->WaveHeightFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"WindSpeedFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->WindSpeedFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"WindDirectionFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->WindDirectionFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"WaveVelocityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->WaveVelocityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"HorizontalVelocityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->HorizontalVelocityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"VerticalVelocityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->VerticalVelocityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"SalinityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->SalinityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"BGSalinityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->BGSalinityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"TemperatureFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->TemperatureFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  MPI_GetFile(filename,DATAFILE,"PressureFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->PressureFID = MPI_FOpen(str,"w","OpenFiles",myproc);
  
  MPI_GetFile(filename,DATAFILE,"EddyViscosityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->EddyViscosityFID = MPI_FOpen(str,"w","OpenFiles",myproc);
  
  
  MPI_GetFile(filename,DATAFILE,"ScalarDiffusivityFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->ScalarDiffusivityFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  /*
  MPI_GetFile(filename,DATAFILE,"VerticalGridFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->VerticalGridFID = MPI_FOpen(str,"w","OpenFiles",myproc);
  */
  MPI_GetFile(filename,DATAFILE,"StoreFile","OpenFiles",myproc);
  sprintf(str,"%s.%d",filename,myproc);
  prop->StoreFID = MPI_FOpen(str,"w","OpenFiles",myproc);

  if(RESTART) {
    MPI_GetFile(filename,DATAFILE,"StartFile","OpenFiles",myproc);
    sprintf(str,"%s.%d",filename,myproc);
    prop->StartFID = MPI_FOpen(str,"r","OpenFiles",myproc);
  }
  /*
  if(myproc==0) {
    MPI_GetFile(filename,DATAFILE,"ConserveFile","OpenFiles",myproc);
    sprintf(str,"%s",filename);
    prop->ConserveFID = MPI_FOpen(str,"w","OpenFiles",myproc);
  }*/
}

/*
 * Function: Progress
 * Usage: Progress(prop,myproc);
 * -----------------------------
 * Output the progress of the calculation to the terminal.
 *
 */
static void Progress(propT *prop, int myproc) 
{
  int progout, prog;
  char filename[BUFFERLENGTH];
  FILE *fid;
  
  MPI_GetFile(filename,DATAFILE,"ProgressFile","Progress",myproc);

  if(myproc==0) {
    fid = fopen(filename,"w");
    fprintf(fid,"On %d of %d, t=%.2f (%d%% Complete, %d output)",
	    prop->n,prop->nstart+prop->nsteps,prop->rtime,100*(prop->n-prop->nstart)/prop->nsteps,
	    1+(prop->n-prop->nstart)/prop->ntout);
    fclose(fid);
  }

  if(myproc==0 && prop->ntprog>0 && VERBOSE>0) {
    progout = (int)(prop->nsteps*(double)prop->ntprog/100);
    prog=(int)(100.0*(double)(prop->n-prop->nstart)/(double)prop->nsteps);
    if(progout>0)
      if(!(prop->n%progout))
	printf("%d%% Complete.\n",prog);
  }
}

/*
 * Function: InterpToFace
 * Usage: uface = InterpToFace(j,k,phys->uc,u,grid);
 * -------------------------------------------------
 * Linear interpolation of a Voronoi-centered value to the face, using the equation
 * 
 * uface = 1/Dj*(def1*u2 + def2*u1);
 *
 * Since some cells may be degenerate, it is safer to use the following:
 *
 * uface = 1/(def1+def2)*(def1*u2 + def2*u1);
 *
 * If the triangle is a right triangle, then first-order upwinded interpolation
 * is used.
 *
 */
static REAL InterpToFace(int j, int k, REAL **phi, REAL **u, gridT *grid) {
  int nc1, nc2;
  REAL def1, def2, Dj;
  nc1 = grid->grad[2*j];
  nc2 = grid->grad[2*j+1];
  Dj = grid->dg[j];
  def1 = grid->def[nc1*NFACES+grid->gradf[2*j]];
  def2 = grid->def[nc2*NFACES+grid->gradf[2*j+1]];

  if(def1==0 || def2==0)
    return UpWind(u[j][k],phi[nc1][k],phi[nc2][k]);
  else    
    return (phi[nc1][k]*def2+phi[nc2][k]*def1)/(def1+def2);
}

/*
 * Function: LagraTracing
 * Usage:
 * -------------------------------------------------
 * Tracing back from a given point, following the Lagrangian trajectory.
 * ii,kk give the start cell number and vertical level
 * xs, zs are input as initial location, and updated to be the trace back location 
 * Horizontal tracing is not allowed to be more than 5 cells.
 */
static void LagraTracing(gridT *grid, physT *phys, propT *prop, int start_cell, int start_layer, int *ii, int *kk, REAL *xs,REAL *zs,int myproc)
{
  int nc,nc1,nc2,i,i2,j,nf,crossing_edge,mem=0,iterx,iterz;
  int jp[]={1,2,0}, jm[]={2,0,1},imem[5],kmem[5];
  REAL us[2],al[3],da[3],vtx[3],vty[3],tres,tzres,dtx,dtz,ws,dtmin,dtxmin;
  
  int debuginterpvelo=0;
 
  *ii=start_cell;
  *kk=start_layer;
  i2=start_cell;
  tres=prop->dt;    //Horizontal tracing
  dtmin=tres/4;
  //dtxmin=dt;
  iterx=0;

  while(tres>dtmin && iterx<5){ 
   
    InterpVelo(grid,phys,*ii, *kk,xs,us,al,vtx,vty,debuginterpvelo);
    crossing_edge=-1;
    dtx=prop->dt;
   
    for(nf=0;nf<NFACES;nf++){
      da[nf]=us[0]*(vty[jm[nf]]-vty[jp[nf]])+us[1]*(vtx[jp[nf]]-vtx[jm[nf]]);
   
      if(al[nf]+tres*da[nf]<0 && da[nf]!=0)
        al[nf]=-al[nf]/da[nf];
      else
        al[nf]=prop->dt+1.0;
      //dtxmin=Min(dtxmin,al[nf]);
      dtx=Min(dtx,al[nf]);
      //if (al[nf]<dt && al[nf]<=dtxmin)
      if(al[nf]<prop->dt)
        crossing_edge=nf;
    }
    
    // compute vertical tracing 
    tzres=dtx;      //Vertical tracing
    iterz=0;
    while (tzres>dtmin && iterz<5){          //zs is the distance from layer k
      
      if(grid->dzz[*ii][*kk]>0){
      	if((*zs)>grid->dzz[*ii][*kk])  *zs=grid->dzz[*ii][*kk];
      	if((*zs)<0)  *zs=0;
        ws = ((grid->dzz[*ii][*kk]-(*zs))*phys->w[*ii][*kk]+(*zs)*phys->w[*ii][*kk+1])/grid->dzz[*ii][*kk];
        if(ws==0)
          tzres=0;
        else{
          dtz= tzres;
       
          if(tzres*ws>(grid->dzz[*ii][*kk]-(*zs)))  dtz = (grid->dzz[*ii][*kk]-(*zs))/ws;
          if(tzres*ws<-(*zs))  dtz = -(*zs)/ws;
        
          if(dtz<tzres){
            if(ws<0 && phys->w[*ii][*kk]<0){
              if(*kk-1>=grid->ctop[*ii]){
                *kk-=1;
                (*zs)=grid->dzz[*ii][*kk];
              }else
                break;
            }
            else if(ws>0 && phys->w[*ii][*kk+1]>0){
              if(*kk+1<grid->Nk[*ii]){
                *kk+=1;
                (*zs)=0;
              }
              else
                break;
            }
            else
              break;
          }
          else  //dtz>=tzres
            (*zs) += dtz*ws;  //zs is the distance from top face. So when ws>0, should be +=.
          tzres = tzres-dtz;
        }  //ws!=0 
      }else
        tzres=0;
      iterz=iterz+1;
    } //End Vertical tracing
    if(iterz==3)  printf("Warning! Vertical tracing crossed more than 2 layers. step: %d, proc: %d, cell: %d, layer: %d \n",
     	prop->n,myproc, start_cell, start_layer);

    // Compute Horizontal Tracing
    xs[0] -= dtx*us[0]; 
    xs[1] -= dtx*us[1];
   
    if(crossing_edge==-1) return;
    
    i2 = *ii;
    j=grid->face[*ii*NFACES+jp[crossing_edge]];
    nc=grid->neigh[*ii*NFACES+jp[crossing_edge]];
    
    // check neighbour top and bottom
    if((*kk>=grid->ctop[nc]) && (*kk<grid->Nk[nc])&& (grid->normal[*ii*NFACES+jp[crossing_edge]]*phys->u[j][*kk]<0))
      *ii=nc;
    else return;
   
    if(dtx>dtmin)
       mem = 0;
    else{
      for(nc=1;nc<=mem;nc++)
        if(imem[nc] ==*ii && kmem[nc] == *kk)
          return;

      mem = mem+1;
      imem[mem] = *ii;
      kmem[mem] = *kk;
    }
    tres = tres-dtx;
    if(*ii==-1 ||grid->ctop[*ii]>=grid->Nk[*ii]){
      *ii = i2;
      return;
    }
    iterx=iterx+1;
 }  //End horizontal tracing

 if(iterx==2)  printf("Warning! Horizontal tracing crossed more than 2 cells. cell: %d, layer: %d \n", start_cell, start_layer);
 if(*ii==-1)*ii = i2;
 sync();
}


static void InterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us, REAL *al, REAL *vtx, REAL *vty,int debugvelo)
{
  int nf,ne1,ne2;
  int jp[]={1,2,0}, jm[]={2,0,1};
  REAL sum,det[3],un[3],ux[3],uy[3],ptx[3],pty[3];

  // compute vtx vty
  for(nf=0;nf<NFACES;nf++){
    vtx[nf]=grid->xp[grid->cells[i*NFACES+nf]];
    vty[nf]=grid->yp[grid->cells[i*NFACES+nf]];
  }	
 
  // compute ux uy
  for(nf=0;nf<NFACES;nf++){
    ne1=grid->face[i*NFACES+nf];
    ne2=grid->face[i*NFACES+jm[nf]];
    ux[nf]=0.0;
    uy[nf]=0.0;
   
    det[nf]=grid->n1[ne1]*grid->n2[ne2]-grid->n1[ne2]*grid->n2[ne1];
    un[nf]=phys->u[ne1][k];
    un[jm[nf]]=phys->u[ne2][k];
    
    if(k>=grid->Nke[ne1] && grid->Nke[ne2]-grid->etop[ne2]>1)
      un[nf]=phys->u[ne1][grid->Nke[ne1]-1];
    if(k>=grid->Nke[ne2] && grid->Nke[ne2]-grid->etop[ne2]>1)
      un[jm[nf]]=phys->u[ne2][grid->Nke[ne2]-1];
    
    ux[nf]=(grid->n2[ne2]*un[nf]-grid->n2[ne1]*un[jm[nf]])/det[nf];
    uy[nf]=(grid->n1[ne1]*un[jm[nf]]-grid->n1[ne2]*un[nf])/det[nf];
  }
    
  //compute ptx[3] pty[3]
  for(nf=0;nf<NFACES;nf++){
    ptx[0]=xs[0];
    pty[0]=xs[1];
    ptx[1]=vtx[jp[nf]];
    pty[1]=vty[jp[nf]];
    ptx[2]=vtx[jm[nf]];
    pty[2]=vty[jm[nf]];
    
    al[nf]=GetArea(ptx,pty,NFACES);
  }   

  // compute us[2] 
  us[0]=0;
  us[1]=0;
  sum=0;
  for(nf=0;nf<NFACES;nf++){
    us[0]+=al[nf]*ux[nf];
    us[1]+=al[nf]*uy[nf];
    sum+=al[nf];
  }

  if(sum!=0){
    us[0]/=sum;
    us[1]/=sum;
  }else
    printf("Error in interpvelo, cell: %d, layer: %d, x: %e, y: %e, area sum=0 \n",i,k,xs[0],xs[1]);

}


static void StableInterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us)
{
  int nf,ne,j,nc;
  int jp[]={1,2,0}, jm[]={2,0,1};
  REAL sum,den[3],un[3],ux[3],uy[3],ptx[3],pty[3],vtx[3],vty[3],prj[3],ix[3],iy[3],ui[2],inx[3],iny[3],inus[2],inxs[2];
  REAL al[3],inal[3];

  for(nf=0;nf<NFACES;nf++){
    vtx[nf]=grid->xp[grid->cells[i*NFACES+nf]];
    vty[nf]=grid->yp[grid->cells[i*NFACES+nf]];
  }

  for(nf=0;nf<NFACES;nf++){
    j=grid->face[i*NFACES+nf];
    nc=grid->neigh[i*NFACES+nf];
    prj[nf]=(vtx[jp[nf]]-vtx[nf])*(xs[0]-vtx[nf])+(vty[jp[nf]]-vty[nf])*(xs[1]-vty[nf]);
    ix[nf]=vtx[nf]+prj[nf]*(vtx[jp[nf]]-vtx[nf])/pow(grid->df[j],2);
    iy[nf]=vty[nf]+prj[nf]*(vty[jp[nf]]-vty[nf])/pow(grid->df[j],2);

    if((ix[nf]<Min(vtx[nf],vtx[jp[nf]])&& vtx[nf]<=vtx[jp[nf]]) ||
         (ix[nf]>Max(vtx[nf],vtx[jp[nf]])&& vtx[nf]>=vtx[jp[nf]])||
         (iy[nf]<Min(vty[nf],vty[jp[nf]])&& vty[nf]<=vty[jp[nf]])||
         (iy[nf]>Max(vty[nf],vty[jp[nf]])&& vty[nf]>=vty[jp[nf]])){
      ix[nf] = vtx[nf];
      iy[nf] = vty[nf];
    }

    if((ix[nf]<Min(vtx[nf],vtx[jp[nf]])&& vtx[nf]>vtx[jp[nf]]) ||
         (ix[nf]>Max(vtx[nf],vtx[jp[nf]])&& vtx[nf]<vtx[jp[nf]])||
         (iy[nf]<Min(vty[nf],vty[jp[nf]])&& vty[nf]>vty[jp[nf]])||
         (iy[nf]>Max(vty[nf],vty[jp[nf]])&& vty[nf]<vty[jp[nf]])){
      ix[nf] = vtx[jp[nf]];
      iy[nf] = vty[jp[nf]];
    }
    inxs[0]=ix[nf];
    inxs[1]=iy[nf];

    if(nc!=-1 && k<grid->Nk[nc])
      InterpVelo(grid,phys,nc,k ,inxs,inus,inal,inx,iny,0);
    else
      InterpVelo(grid,phys,i,k ,inxs,inus,inal,inx,iny,0);
    
    ux[nf]=inus[0];
    uy[nf]=inus[1];

  }
 
  us[0]=0;
  us[1]=0;
  sum=0;
  for(nf=0;nf<NFACES;nf++){
    ptx[0]=xs[0];
    pty[0]=xs[1];
    ptx[1]=ix[jp[nf]];
    pty[1]=iy[jp[nf]];
    ptx[2]=ix[jm[nf]];
    pty[2]=iy[jm[nf]];
    al[nf]=GetArea(ptx,pty,NFACES);
    us[0]+=al[nf]*ux[nf];
    us[1]+=al[nf]*uy[nf];
    sum+=al[nf];
  }

  if(sum>0){
    us[0]/=sum;
    us[1]/=sum;
  }else
    InterpVelo(grid,phys,i,k ,xs,us,inal,inx,iny,0);

}

/*
 * Function: SetDensity
 * Usage: SetDensity(grid,phys,prop,sprop);
 * ----------------------------------
 * Sets the values of the density in the density array rho and
 * at the boundaries.
 *
 */
static void SetDensity(gridT *grid, physT *phys, sediT *sedi, propT *prop, spropT *sprop) {
  int i, j, k, jptr, ib;
  REAL z, p, sd=0.0;

  for(i=0;i<grid->Nc;i++) {
    z=phys->h[i];
    for(k=grid->ctop[i];k<grid->Nk[i];k++) {
      z+=0.5*grid->dzz[i][k];
      p=RHO0*GRAV*z;      

      if(prop->sedi)
	if(sprop->strati)
	  sd = Min(sedi->sdtot[i][k]/2650000.0, 0.01);
	
      phys->rho[i][k]=StateEquation(prop,phys->s[i][k],phys->T[i][k],sd,p);
      z+=0.5*grid->dzz[i][k];

    }
  }

  for(jptr=grid->edgedist[2];jptr<grid->edgedist[3];jptr++) {
      j=grid->edgep[jptr];
      ib=grid->grad[2*j];

      z=phys->h[ib];
      for(k=grid->ctop[ib];k<grid->Nk[ib];k++) {
	z+=0.5*grid->dzz[ib][k];
	p=RHO0*GRAV*z;
	phys->boundary_rho[jptr-grid->edgedist[2]][k]=
	  StateEquation(prop,phys->boundary_s[jptr-grid->edgedist[2]][k],
			phys->boundary_T[jptr-grid->edgedist[2]][k],
                        0.0, p);
	z+=0.5*grid->dzz[ib][k];
      }
  }
}

void InputTides(gridT *grid, physT *phys, propT *prop, int myproc){

  char str[BUFFERLENGTH], filename[BUFFERLENGTH];
  char tmp[BUFFERLENGTH];
  char c;
  FILE *ifile;
  int n, Ninput, i, j, Noffset;
  double tmp2;

  MPI_GetFile(filename, DATAFILE, "TidalDataFile", "InputTides", myproc);
  ifile = MPI_FOpen(filename, "r", "InputTides", myproc);

  Ninput = 1+prop->nsteps*prop->dt/prop->dt_tideBC;
  Noffset = 2*prop->nstart*prop->dt/prop->dt_tideBC;

  //str[0] = ' ';
  //while (str[0] != '-'){
  //  getline(ifile, str, "...");
  //}
  //  c = fgetc(ifile);

  //  if (c == EOF){
  //    return;
  //  }
  //If this is a continuous run, skip the lines that have been read 
  //from the previous run.
     for(n = 0; n < Noffset; n++){
      getline(ifile, str, "");
    }    
  //   printf("yc prop->nstart = %d", Noffset);
  //c = fgetc(ifile);

  //if (c == EOF){
  //  printf("End of the tide data file is encountered\n");  
  //  return;
  //}

  for(n = 0; n < Ninput; n++){
    fscanf(ifile,"%lf", &tmp2);
    phys->wl_tideBC[n] = tmp2;
//    printf("tide data= %lf\n",phys->wl_tideBC[n]);
/*
    i = 0;
    while (c != '\n' & c!= EOF & c != '\t'){
      c = fgetc(ifile);
      i++;
      if (i >= 32 & i <= 38)
	str[i-32] = c;
    }
    
    if (str[3] == ' '){
      if (n == 0){
	for (j = 0; j< 38-32+1; j++)
	  str[j] = '0';
      }else{
	for (j = 0; j< 38-32+1; j++)
	  str[j] = tmp[j];
      }
    }

    for (j = 0; j< 38-32+1; j++)
      tmp[j] = str[j];

    phys->wl_tideBC[n] = strtod(str, (char **)NULL);
    c = fgetc(ifile);
*/
  }
  fclose(ifile);

}

