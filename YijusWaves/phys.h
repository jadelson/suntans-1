/*
 * File: phys.h
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * Header file for phys.c.
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior               
 * University. All Rights Reserved.
 *
 */
#ifndef _phys_h
#define _phys_h

#include "suntans.h"
#include "grid.h"
#include "fileio.h"

/*
 * Main physical variable struct.
 *
 */
typedef struct _physT {
  REAL **u;
  REAL **uc;
  REAL **vc;
  REAL **tau_SD;
  REAL **uold;
  REAL **vold;
  REAL *D;
  REAL **w;
  REAL **wf;
  REAL **q;
  REAL **qc;
  REAL **s;
  REAL **T;
  REAL **s0;
  REAL **rho;
  REAL *h;
  REAL *de;  

  REAL **boundary_u;
  REAL **boundary_v;
  REAL **boundary_w;
  REAL **boundary_s;
  REAL **boundary_T;
  REAL **boundary_tmp;
  REAL **boundary_rho;
  REAL *boundary_h;
  REAL *boundary_flag;

  REAL **nu_tv;
  REAL **kappa_tv;
  REAL *tau_T;
  REAL *tau_B;
  REAL *CdT;
  REAL *CdB;
  REAL **qT;
  REAL **lT;

  REAL mass;
  REAL mass0;
  REAL volume;
  REAL volume0;
  REAL Ep;
  REAL Ep0;
  REAL Eflux1;
  REAL Eflux2;
  REAL Eflux3;
  REAL Eflux4;
  REAL smin;
  REAL smax;

  REAL *htmp;
  REAL *hold;
  REAL **stmp;
  REAL **stmp2;
  REAL **stmp3;
  REAL **utmp;
  REAL **utmp2;

  REAL **ut;
  REAL **Cn_R;
  REAL **Cn_T;
  REAL **Cn_U;
  REAL **Cn_W;
  REAL **Cn_q;
  REAL **Cn_l;
  REAL **wtmp;
  REAL **wtmp2;
  REAL **qtmp;

  REAL *ap;
  REAL *am;
  REAL *bp;
  REAL *bm;

  REAL *a;
  REAL *b;
  REAL *c;
  REAL *d;
  
  // Horizontal facial scalar
  REAL **SfHp;
  REAL **SfHm;

  //Define variables for TVD schemes
  REAL *Cp;
  REAL *Cm;
  REAL *rp;
  REAL *rm;
  REAL *wp;
  REAL *wm;
  REAL **gradSx;
  REAL **gradSy;

  REAL **tvdminus;
  REAL **tvdplus;
  REAL **rterm;
  REAL **rterm2;
  REAL **udfsminus;
  REAL **udfminus;
  REAL *udfs;
  REAL *udf;
  REAL *ustvd;
  REAL *ustvd_minus;
  REAL *ustvd_plus;

  // Vivien
  // read salinity
  REAL *xsal;
  REAL *ysal;
  REAL *zsal;
  REAL *sal;

  // Yi-Ju
  // for wave modeling
  REAL *dhdt;
  REAL *wl_tideBC;
} physT;

typedef struct _sediT {
  REAL ***sd;
  REAL ***breakup;
  REAL ***aggreg;
  REAL **sdtot;
  
////////// Kurt Nelson Added For Profile Output//////
  REAL **scc0;
  REAL **scc1;
/////////////////////////////////////////////////////


  REAL **G;
  REAL **ws;

  REAL *pd;
  REAL *tau_cd;
  REAL *Cdc;

  REAL **tau_ce;
  REAL **M;

  REAL *Dl;
  REAL *Tcsl;
  REAL *E;
  REAL *alpha;
  REAL *hl;
  REAL *cnsd;

  REAL *z0s;
  REAL *z0b;
  REAL *z0r;
  REAL *kb;

  REAL *tau_w;
  REAL *tau_c;

  REAL *pickup;
  REAL *bedmudratio;
  REAL *dpsit;

} sediT;

typedef struct _waveT {
  REAL *sg;
  REAL *dsg;
  REAL *thtaw;
  REAL **kw;
  REAL *ktail;
  REAL ***cgx;
  REAL ***cgy;
  REAL ***cg;
  REAL **cph;
  REAL ***cs;
  REAL ***ct;
  REAL ***N;
  REAL ***Nold;
  REAL ***Ntmp;
  REAL ***src;
  REAL *Etot;
  REAL *Etmp;
  REAL *kmean;
  REAL *sgmean;
  REAL *T0;
  REAL *Hs;
  REAL *T01;
  REAL *sg_PM;
  REAL *thtamean;
  REAL *Cr;
  REAL *fw;
  REAL *uscx;
  REAL *uscy;
  REAL *use;
  REAL *Uwind;
  REAL *Etail;
  REAL *Ntail;

  REAL *a;
  REAL *b;
  REAL *c;

  REAL *sp;
  REAL *sm;
  REAL ***ssrc;

  REAL *tp;
  REAL *tm;
  REAL ***tsrc;

  REAL **wind_sp;
  REAL **wind_dg;
  REAL **klambda;

  REAL *wind_spfx;
  REAL *wind_spfy;
  REAL *wind_spf;
  REAL *wind_dgf;

  REAL *Hw;
  REAL *ub;
  REAL *ab;
  REAL *tmp;
  REAL **fz;
  REAL *fphi;

  REAL **ux;
  REAL **uy;
  REAL **uz;  
  REAL **Uw;
  REAL **divScx;
  REAL **divScy;
  REAL **divSe;
  
  REAL *ab_edge;
  REAL *sgmean_edge;
  REAL *thtamean_edge;
  REAL *kmean_edge;

  REAL **kw_edge;
} waveT;

/*
 * Main sediment property struct
 *
 */
typedef struct _spropT {

  REAL ws0, Cfloc, Chind, gamma, diam, spwght, k, Kb;  
  REAL Kagg, Kbrk, Fy, p, q, nf;
  REAL diam_min, diam_max, delta_diam, Dp, vis;
  REAL *diam_repr, *wss, *btm_partition;
  int NL;
  int Nsize;
  int strati;
  int size_exchange; 

} spropT;

/*
 * Main property struct.
 *
 */
typedef struct _wpropT {

  int Mw, Nw;
  int wnstep;
  REAL sgmin;
  REAL sgmax;
  REAL sg99;
  REAL sgtail;
  REAL wind_dt;
  int implicit_whitecap;
  int implicit_advection;
  int wind_forcing;
  int Nwind, nwind;
  int nstation;
  int wind_shear;
  int rad_stress;
  int form_drag;
  int btm_mud;
  int btm_sedi_erosion;
  REAL depth_fw_cutoff;
  REAL fw_drag;
  REAL *xw, *yw;
  REAL **lambda;
  REAL btm_conc;
  REAL btm_vis;
  REAL btm_mud_thickness;
  int depth_brk_cutoff;
  REAL depth_brk_indx;

  int tail_opt;
  REAL tail_pow;
  int NLtriad;
  int NLquad;
  int BRKdepth;

} wpropT;

/*
 * Main property struct.
 *
 */
typedef struct _propT {
  REAL dt, Cmax, rtime, amp, omega, flux, timescale, theta0, theta, 
    thetaS, thetaB, nu, nu_H, tau_Tp, z0T, CdT, z0B, CdB, CdW, relax, epsilon, qepsilon, resnorm, 
    dzsmall, beta, kappa_s, kappa_sH, gamma, kappa_T, kappa_TH, Coriolis_f, dt_tideBC;
  int ntout, ntprog, nsteps, nstart, n, ntconserve, nonhydrostatic, cgsolver, maxiters, qmaxiters, qprecond, volcheck, masscheck,
    nonlinear, newcells, wetdry, sponge_distance, sponge_decay, thetaramptime, readSalinity, readTemperature, turbmodel, TVD, 
    wave, sedi, horiTVD, vertTVD;
  FILE *FreeSurfaceFID, *HorizontalVelocityFID, *VerticalVelocityFID,
    *SalinityFID, *BGSalinityFID, *InitSalinityFID, *InitTemperatureFID, *TemperatureFID, *PressureFID, *VerticalGridFID, *ConserveFID,
    *StoreFID, *StartFID, *EddyViscosityFID, *ScalarDiffusivityFID, *WindSpeedFID, *WindDirectionFID, *WaveVelocityFID, *WaveHeightFID,
    *SediDepositionFID, *SSCFID, *SSCOneFID, *SSCZeroFID;
} propT;




/*
 * Public function declarations.
 *
 */
void Solve(gridT *grid, physT *phys, propT *prop, spropT *sprop, sediT *sedi,
           waveT *wave, wpropT *wprop, int myproc, int numprocs, MPI_Comm comm);
void AllocatePhysicalVariables(gridT *grid, physT **phys, propT *prop);
void FreePhysicalVariables(gridT *grid, physT *phys, propT *prop);
void InitializePhysicalVariables(gridT *grid, physT *phys, sediT *sedi, propT *prop, spropT *sprop);
void InitializeVerticalGrid(gridT **grid);
void ReadPhysicalVariables(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, 
			   waveT *wave, wpropT *wprop, int myproc, MPI_Comm comm);
void OpenFiles(propT *prop, int myproc);
void ReadProperties(propT **prop, int myproc);
void SetDragCoefficients(gridT *grid, physT *phys, propT *prop);
void InputTides(gridT *grid, physT *phys, propT *prop, int myproc);

#endif
