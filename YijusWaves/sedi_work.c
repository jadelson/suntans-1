/*
 * File: sedi.c
 * Author: Yi-Ju Chou
 */

#include "sedi.h"
#include "util.h"
#include "tvd.h"
#include "mympi.h"
#include "memory.h"
#include "turbulence.h"

static void SizeExchange(gridT *grid, sediT *sedi, spropT *sprop, propT *prop, int sdindx, MPI_Comm comm, int myproc);

void InitializeSediProperties(propT *prop, spropT **sprop, int myproc)
{
  int i, Ns;
  REAL rho_floc, tmp, tmp2;

  *sprop = (spropT *)SunMalloc(sizeof(spropT), "InitializeSediProperties");
  
  (*sprop)->NL = MPI_GetValue(SEDIFILE,"NL","InitializeSediProperties",myproc); 
  (*sprop)->spwght = MPI_GetValue(SEDIFILE,"spwght","InitializeSediProperties",myproc);
  (*sprop)->diam = MPI_GetValue(SEDIFILE,"diam","InitializeSediProperties",myproc);
  (*sprop)->gamma = MPI_GetValue(SEDIFILE,"gamma","InitializeSediProperties",myproc);
  (*sprop)->Chind = MPI_GetValue(SEDIFILE,"Chind","InitializeSediProperties",myproc);
  (*sprop)->Cfloc = MPI_GetValue(SEDIFILE,"Cfloc","InitializeSediProperties",myproc);
  (*sprop)->k = MPI_GetValue(SEDIFILE,"k","InitializeSediProperties",myproc);
  (*sprop)->Kb = MPI_GetValue(SEDIFILE,"Kb","InitializeSediProperties",myproc);
  (*sprop)->Kagg = MPI_GetValue(SEDIFILE,"Kagg","InitializeSediProperties",myproc);
  (*sprop)->Kbrk = MPI_GetValue(SEDIFILE,"Kbrk","InitializeSediProperties",myproc);
  (*sprop)->Fy = MPI_GetValue(SEDIFILE,"Fy","InitializeSediProperties",myproc);
  (*sprop)->nf = MPI_GetValue(SEDIFILE,"nf","InitializeSediProperties",myproc);
  (*sprop)->q = MPI_GetValue(SEDIFILE,"q","InitializeSediProperties",myproc);
  (*sprop)->p = 3-(*sprop)->nf;
  (*sprop)->Nsize = MPI_GetValue(SEDIFILE,"Nsize","InitializeSediProperties",myproc);
  (*sprop)->size_exchange = MPI_GetValue(SEDIFILE,"size_exchange","InitializeSediProperties",myproc);
  (*sprop)->diam_min = MPI_GetValue(SEDIFILE,"diam_min","InitializeSediProperties",myproc);
  (*sprop)->diam_max = MPI_GetValue(SEDIFILE,"diam_max","InitializeSediProperties",myproc);
  (*sprop)->Dp = MPI_GetValue(SEDIFILE,"Dp","InitializeSediProperties",myproc);
  (*sprop)->vis = MPI_GetValue(SEDIFILE,"vis","InitializeSediProperties",myproc);
  (*sprop)->strati = MPI_GetValue(SEDIFILE,"strati","InitializeSediProperties",myproc);

  Ns = (*sprop)->Nsize;
  (*sprop)->delta_diam = ((*sprop)->diam_max-(*sprop)->diam_min)/Ns;
  (*sprop)->diam_repr = (REAL *)SunMalloc(Ns*sizeof(REAL), "InitializeSediProperties");
  (*sprop)->wss = (REAL *)SunMalloc(Ns*sizeof(REAL), "InitializeSediProperties");
  (*sprop)->btm_partition = (REAL *)SunMalloc(Ns*sizeof(REAL), "InitializeSediProperties");

  //  (*sprop)->diam_repr[0] = (*sprop)->diam_min + 0.5*(*sprop)->delta_diam;
  (*sprop)->diam_repr[0] = (*sprop)->diam_min;
  for (i = 1; i < Ns; i++){
    //    (*sprop)->diam_repr[i] = (*sprop)->diam_repr[i-1] + (*sprop)->delta_diam;
    (*sprop)->diam_repr[i] = (*sprop)->diam_max;
  }
  for (i = 0; i < Ns; i++){
    rho_floc = RHO0 + ((*sprop)->spwght-1)*RHO0*pow((*sprop)->Dp/(*sprop)->diam_repr[i], 3-(*sprop)->nf);
    (*sprop)->wss[i] = -(rho_floc - RHO0)*GRAV*pow((*sprop)->diam_repr[i], 2)/(18*prop->nu*RHO0);
  }
  (*sprop)->wss[0] = MPI_GetValue(SEDIFILE,"ws_mic","InitializeSediProperties",myproc);
  (*sprop)->wss[1] = MPI_GetValue(SEDIFILE,"ws_mac","InitializeSediProperties",myproc);

  if(Ns == 2){
      (*sprop)->btm_partition[0] = MPI_GetValue(SEDIFILE,"btm_partition_mic","InitializeSediProperties",myproc);
      (*sprop)->btm_partition[1] = 1.0-(*sprop)->btm_partition[0];
  }else{
      (*sprop)->btm_partition[0] = 1;
  }

}


void AllocateSediVariables(gridT *grid, sediT **sedi, spropT *sprop, propT *prop)
{
  int flag=0, i, j, Nc=grid->Nc, NL=sprop->NL, Ns=sprop->Nsize, s;

  *sedi = (sediT *)SunMalloc(sizeof(sediT), "AllocateSediVariables");
 
  (*sedi)->sd = (REAL ***)SunMalloc(Ns*sizeof(REAL **), "AllocateSediVariables");
  for (s=0; s<Ns; s++){
    (*sedi)->sd[s] = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
    for (i=0; i<Nc; i++){
      (*sedi)->sd[s][i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL), "AllocateSediVariables");
    }
  }
  (*sedi)->breakup = (REAL ***)SunMalloc(Ns*sizeof(REAL **), "AllocateSediVariables");
  (*sedi)->aggreg = (REAL ***)SunMalloc(Ns*sizeof(REAL **), "AllocateSediVariables");
  for (s=0; s<Ns; s++){
    (*sedi)->breakup[s] = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
    (*sedi)->aggreg[s] = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
    for (i=0; i<Nc; i++){
      (*sedi)->breakup[s][i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL), "AllocateSediVariables");
      (*sedi)->aggreg[s][i] = (REAL *)SunMalloc(grid->Nk[i]*sizeof(REAL), "AllocateSediVariables");
    }
  }
  (*sedi)->G  = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
  (*sedi)->sdtot  = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
//Kurt Nelson added the two varibles below for output in profiles.c
  (*sedi)->scc0  = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
  (*sedi)->scc1  = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
///////////////////////////////////////////////////////////////////////////////
  (*sedi)->ws = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
  (*sedi)->z0s = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->z0b = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->z0r = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->kb  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->tau_w  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->tau_c  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->pickup  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->bedmudratio  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->dpsit  = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");

  for (i=0; i<Nc; i++){
    (*sedi)->ws[i] = (REAL *)SunMalloc((grid->Nk[i]+1)*sizeof(REAL), "AllocateSediVariables");
    (*sedi)->G[i] = (REAL *)SunMalloc((grid->Nk[i])*sizeof(REAL), "AllocateSediVariables");
    (*sedi)->sdtot[i] = (REAL *)SunMalloc((grid->Nk[i])*sizeof(REAL), "AllocateSediVariables");
//Kurt Nelson added the two varibles below for output in profiles.c
   (*sedi)->scc0[i] = (REAL *)SunMalloc((grid->Nk[i])*sizeof(REAL), "AllocateSediVariables");
   (*sedi)->scc1[i] = (REAL *)SunMalloc((grid->Nk[i])*sizeof(REAL), "AllocateSediVariables");
/////////////////////////////////////////////////////////////////////////////////////////////////////
  }
  (*sedi)->pd = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->tau_cd = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");

  (*sedi)->tau_ce = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
  (*sedi)->M = (REAL **)SunMalloc(Nc*sizeof(REAL *), "AllocateSediVariables");
  (*sedi)->Cdc = (REAL *)SunMalloc(Nc*sizeof(REAL), "AllocateSediVariables");
  for (i=0; i<Nc; i++){
    (*sedi)->tau_ce[i] = (REAL *)SunMalloc(5*sizeof(REAL), "AllocateSediVariables");    
    (*sedi)->M[i] = (REAL *)SunMalloc(5*sizeof(REAL), "AllocateSediVariables");
  }
  (*sedi)->Dl = (REAL *)SunMalloc(5*sizeof(REAL), "AllocateSediVariables");   
  (*sedi)->Tcsl = (REAL *)SunMalloc(5*sizeof(REAL), "AlloacteSediVariables");
  (*sedi)->cnsd = (REAL *)SunMalloc(5*sizeof(REAL), "AlloacteSediVariables");
  (*sedi)->E = (REAL *)SunMalloc(5*sizeof(REAL), "AllocateSediVariables");
  (*sedi)->alpha = (REAL *)SunMalloc(5*sizeof(REAL), "AllocateSediVariables");   
  (*sedi)->hl = (REAL *)SunMalloc(6*sizeof(REAL), "AllocateSediVariables");   

}


void FreeSediVariables(gridT *grid, sediT *sedi, propT *prop, spropT *sprop)
{
  int s, i, j, Nc=grid->Nc, NL=sprop->NL, Ns = sprop->Nsize;
  
  for(i=0;i<Nc;i++){
    free(sedi->ws[i]);
    free(sedi->G[i]);
    free(sedi->sdtot[i]);
//Kurt Nelson added
    free(sedi->scc0[i]);
    free(sedi->scc1[i]);
/////////////////////////////////////////////////
    free(sedi->tau_ce[i]);
    free(sedi->M[i]);

  }

  for(s=0; s < Ns; s++){
      for(i=0; i<Nc; i++){
	free(sedi->sd[s][i]);
      }
      free(sedi->sd[s]);
  }

  for(s=0; s < Ns; s++){
      for(i=0; i<Nc; i++){
	free(sedi->aggreg[s][i]);
	free(sedi->breakup[s][i]);
      }
      free(sedi->aggreg[s]);
      free(sedi->breakup[s]);
  }


  free(sedi->sd);
  free(sedi->aggreg);
  free(sedi->breakup);
  free(sedi->ws);
  free(sedi->G);
  free(sedi->sdtot);
//Kurt Nelson added
  free(sedi->scc0);
  free(sedi->scc1);
//////////////////////////////////////////////
  free(sedi->pd);
  free(sedi->tau_cd);
  free(sedi->tau_ce);
  free(sedi->Cdc);
  free(sedi->M);

  free(sedi->Dl);
  free(sedi->Tcsl);
  free(sedi->cnsd);
  free(sedi->E);
  free(sedi->alpha);
  free(sedi->hl);
 
  free(sedi->z0s);
  free(sedi->z0b);
  free(sedi->z0r);
  free(sedi->kb);
  free(sedi->tau_w);
  free(sedi->tau_c);
  free(sedi->pickup);
  free(sedi->bedmudratio);
  free(sedi->dpsit);

  free(sedi);

}

//void UpdateOrbitalVel(sediT *sedi, spropT *sprop, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs){

//  int i, Nc=grid->Nc, N=floor(prop->rtime/wprop->wind_dt);
//  REAL fetch, depth, wsp, gus, fgus, dgus, wl, D, tmp;

//  for(i=0; i< Nc; i++){
//    fetch = sqrt(pow(grid->xv[i]-sedi->xw1, 2)
//	       + pow(grid->yv[i]-sedi->yw1, 2));
//   depth = phys->h[i]+grid->dv[i];
//    if (depth < 0.1) depth = 0.1;
//    wsp = sedi->wind_sp[N];
//   if (wsp == 0) wsp = 0.0001;
//    gus = GRAV/pow(wsp,2);
//    fgus = fetch*gus;
//    dgus = depth*gus;

//   sedi->Hw[i] = 1/gus*0.283*tanh(0.53*pow(dgus, 0.75))*tanh(0.00565*pow(fgus, 0.5)/tanh(0.53*pow(dgus,0.75)));
//    sedi->Tw[i] = 1/gus/wsp*7.54*tanh(0.833*pow(dgus, 0.375))*tanh(0.0379*pow(fgus, 1/3)/tanh(0.8333*pow(dgus,0.375)));
//    D = 4*pow(PI,2)*depth/(GRAV*pow(sedi->Tw[i], 2));
    //    tmp = sqrt(D*(D+1/(1+D*(0.6522+D*(0.4622+pow(D, 2)*(0.0864+0.0675*D))))));
    //tmp = 1/tmp;
//    tmp = 1 + 0.6522*D+0.4622*pow(D, 2)+0.0864*pow(D, 4) + 0.0675*pow(D, 5);
//    tmp = D+1/tmp;
    //    wl = 2*PI*depth*tmp;
//   wl = sedi->Tw[i]*sqrt(GRAV*depth/tmp);
//    sedi->ub[i] = PI*sedi->Hw[i]/(sedi->Tw[i]*sinh(2*PI*depth/wl));
    
    //    printf("i am here ...................................Hw = %f Tw = %f wl = %f\n", sedi->Hw[i], sedi->Tw[i], wl);
//  }

//}


void InitializeSediVariables(gridT *grid, sediT *sedi, propT *prop, spropT *sprop, int myproc, MPI_Comm comm)
{
  int i, k, s, Nc=grid->Nc, NL=sprop->NL, Ns = sprop->Nsize;
  REAL dpth;

  for(s=0; s<Ns; s++ ){
    for(i=0;i<Nc;i++){
      for(k=0;k<grid->Nk[i];k++){
	sedi->sd[s][i][k]=0;
      }
    }
  }
  for(s=0; s<Ns; s++ ){
    for(i=0;i<Nc;i++){
      for(k=0;k<grid->Nk[i];k++){
	sedi->breakup[s][i][k] = 0;
	sedi->aggreg[s][i][k] = 0;
      }
    }
  }
  for(i=0;i<Nc;i++){    
    sedi->pd[i]=0;
    sedi->tau_cd[i]=0.05;
    sedi->Cdc[i] = 0;

    sedi->z0s[i] = 2.5*sprop->diam/30;
    //sedi->z0s[i] = sprop->diam/30;
    sedi->z0b[i] = 0;
    sedi->z0r[i] = 0;
    sedi->kb[i] = 30*sedi->z0s[i];
    sedi->tau_w[i] = 0;
    sedi->tau_c[i] = 0;
    sedi->pickup[i] = 0;
    sedi->dpsit[i] = 0;


    for(k=0;k<grid->Nk[i];k++){
      sedi->ws[i][k]=0;
      sedi->G[i][k]=0;
      sedi->sdtot[i][k] = 0; 
//Kurt Nelson added
      sedi->scc0[i][k] = 0;
      sedi->scc1[i][k] = 0;
///////////////////////////////////////////////////

    }
    sedi->ws[i][grid->Nk[i]]=0;
    for(k=0;k<NL;k++){
      sedi->tau_ce[i][k]=0;
      sedi->M[i][k]=0;
    }
  }
  for(i=0;i<Nc;i++){
    if(grid->yv[i] < 4200000){
      if(grid->yv[i] < 4160000) //Lower South Bay
	sedi->bedmudratio[i] = 0.8;
      else{
	if(grid->yv[i] < 4175000) //South Bay
	  sedi->bedmudratio[i] = 0.5;
	else{
	  dpth = grid->dv[i]; //Central Bay
	  sedi->bedmudratio[i] = 1.0-(0.0514*dpth+0.0796);
	  if(sedi->bedmudratio[i] > 1.0) sedi->bedmudratio[i] = 1.0;
	  if(sedi->bedmudratio[i] < 0.0) sedi->bedmudratio[i] = 0.0;
	}
      }
    }else{
      if(grid->xv[i] < 570000)
	sedi->bedmudratio[i] = 0.8; // San Pablo Bay
      else
	sedi->bedmudratio[i] = 0.5; // Suisun Bay
    }
  }

  sedi->Dl[0] = MPI_GetValue(SEDIFILE,"Dl1","InitializeSediVariables",myproc);
  sedi->Dl[1] = MPI_GetValue(SEDIFILE,"Dl2","InitializeSediVariables",myproc);         
  sedi->Dl[2] = MPI_GetValue(SEDIFILE,"Dl3","InitializeSediVariables",myproc);         
  sedi->Dl[3] = MPI_GetValue(SEDIFILE,"Dl4","InitializeSediVariables",myproc);         
  sedi->Dl[4] = MPI_GetValue(SEDIFILE,"Dl5","InitializeSediVariables",myproc);


  sedi->Tcsl[0] = MPI_GetValue(SEDIFILE,"Tcsl1","InitializeSediVariables",myproc);
  sedi->Tcsl[1] = MPI_GetValue(SEDIFILE,"Tcsl2","InitializeSediVariables",myproc);         
  sedi->Tcsl[2] = MPI_GetValue(SEDIFILE,"Tcsl3","InitializeSediVariables",myproc);         
  sedi->Tcsl[3] = MPI_GetValue(SEDIFILE,"Tcsl4","InitializeSediVariables",myproc);         
  sedi->Tcsl[4] = MPI_GetValue(SEDIFILE,"Tcsl5","InitializeSediVariables",myproc);

  sedi->E[0] = MPI_GetValue(SEDIFILE,"E1","InitializeSediVariables",myproc);
  sedi->E[1] = MPI_GetValue(SEDIFILE,"E2","InitializeSediVariables",myproc);         
  sedi->E[2] = MPI_GetValue(SEDIFILE,"E3","InitializeSediVariables",myproc);         
  sedi->E[3] = MPI_GetValue(SEDIFILE,"E4","InitializeSediVariables",myproc);         
  sedi->E[4] = MPI_GetValue(SEDIFILE,"E5","InitializeSediVariables",myproc);

  sedi->alpha[0] = MPI_GetValue(SEDIFILE,"alpha1","InitializeSediVariables",myproc);
  sedi->alpha[1] = MPI_GetValue(SEDIFILE,"alpha2","InitializeSediVariables",myproc);         
  sedi->alpha[2] = MPI_GetValue(SEDIFILE,"alpha3","InitializeSediVariables",myproc);         
  sedi->alpha[3] = MPI_GetValue(SEDIFILE,"alpha4","InitializeSediVariables",myproc);         
  sedi->alpha[4] = MPI_GetValue(SEDIFILE,"alpha5","InitializeSediVariables",myproc); 

  sedi->cnsd[0] = MPI_GetValue(SEDIFILE,"cnsd1","InitializeSediVariables",myproc);
  sedi->cnsd[1] = MPI_GetValue(SEDIFILE,"cnsd2","InitializeSediVariables",myproc);         
  sedi->cnsd[2] = MPI_GetValue(SEDIFILE,"cnsd3","InitializeSediVariables",myproc);         
  sedi->cnsd[3] = MPI_GetValue(SEDIFILE,"cnsd4","InitializeSediVariables",myproc);         
  sedi->cnsd[4] = MPI_GetValue(SEDIFILE,"cnsd5","InitializeSediVariables",myproc);

  sedi->hl[0] = MPI_GetValue(SEDIFILE,"hl1","InitializeSediVariables",myproc);
  sedi->hl[1] = MPI_GetValue(SEDIFILE,"hl2","InitializeSediVariables",myproc);         
  sedi->hl[2] = MPI_GetValue(SEDIFILE,"hl3","InitializeSediVariables",myproc);         
  sedi->hl[3] = MPI_GetValue(SEDIFILE,"hl4","InitializeSediVariables",myproc);
  sedi->hl[4] = MPI_GetValue(SEDIFILE,"hl5","InitializeSediVariables",myproc);

  for(i=0;i<Nc;i++){
      
    sedi->tau_ce[i][0]=sedi->Tcsl[0];
    sedi->tau_ce[i][1]=sedi->Tcsl[1];
    sedi->tau_ce[i][2]=sedi->Tcsl[2];
    sedi->tau_ce[i][3]=sedi->Tcsl[3];
    sedi->tau_ce[i][4]=sedi->Tcsl[4];
    
  }

  for(i=0;i<Nc;i++){
    for(k=0;k<NL;k++){
      sedi->M[i][k]=sedi->hl[k]*sedi->Dl[k];
    }
  }
  
}

void UpdateSedi(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave, REAL **sd, int sdindx,
                REAL **boundary_scal, REAL **Cn, REAL kappa, REAL kappaH, REAL **kappa_tv, REAL theta,
		REAL **src1, REAL **src2, REAL *Ftop, REAL *Fbot, int alpha_top, int alpha_bot,
		MPI_Comm comm, int myproc)
{
  int i, iptr, j, jptr, ib, k, nf, ktop, indx, kk, indxx;
  int Nc=grid->Nc, normal, nc1, nc2, ne;
  REAL df, dg, Ac, dt=prop->dt, fab, *a, *b, *c, *d, *ap, *am, *bd, dznew, mass, *sp, *temp,
    Cdavg, U_sq, taub, cb, wsc, Uf, pec, tau_e, angle, C0;


  prop->TVD = TVDMACRO;
  // These are used mostly debugging to turn on/off vertical and horizontal TVD.
  prop->horiTVD = 1;
  prop->vertTVD = 1;
  C0 = sedi->Dl[0];

  ap = phys->ap;
  am = phys->am;
  bd = phys->bp;
  temp = phys->bm;
  a = phys->a;
  b = phys->b;
  c = phys->c;
  d = phys->d;
  
  //ObtainSettlingVelocity(grid, phys, sedi, sprop);     
  // Don't use AB2 when wetting and drying is employed or if
  // using the TVD schemes.
  if(prop->n==1 || prop->wetdry || prop->TVD!=0) {
    fab=1;
    for(i=0;i<grid->Nc;i++)
      for(k=0;k<grid->Nk[i];k++)
	Cn[i][k]=0;
  } else
    fab=1.5;

    for(i=0;i<Nc;i++) 
      for(k=0;k<grid->Nk[i];k++) 
	phys->stmp[i][k]=sd[i][k];

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
  
  // Compute the scalar on the vertical faces (for horiz. advection)
  if(prop->TVD && prop->horiTVD)
    HorizontalFaceScalars(grid,phys,prop,boundary_scal,prop->TVD,comm,myproc); 

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++){
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

    for(k=0;k<grid->Nk[i]+1;k++) {
      phys->w[i][k]+=sprop->wss[sdindx];
    }
    // These are the advective components of the tridiagonal
    // at the new time step.
    if(!(prop->TVD && prop->vertTVD))
      
      for(k=0;k<grid->Nk[i]+1;k++) {
	ap[k] = 0.5*(phys->w[i][k]+fabs(phys->w[i][k]));
	am[k] = 0.5*(phys->w[i][k]-fabs(phys->w[i][k]));
	}
    else  // Compute the ap/am for TVD schemes
      GetApAm(ap,am,phys->wp,phys->wm,phys->Cp,phys->Cm,phys->rp,phys->rm,
	      phys->w,grid->dzz,sd,i,grid->Nk[i],ktop,prop->dt,prop->TVD);
      
    for(k=0;k<grid->Nk[i]+1;k++) {
      phys->w[i][k]-=sprop->wss[sdindx];
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
    //   b[(grid->Nk[i]-1)-ktop]+=c[(grid->Nk[i]-1)-ktop];
    // Change for sediment BC. 
    // Since the flux through the bottom boundary is represented by pickup,
    // no vertical velocity at the bottom boundary is considered in the solver. 

      //      b[(grid->Nk[i]-1)-ktop]+=theta*dt*am[grid->Nk[i]-ktop];
      //      c[(grid->Nk[i]-1)-ktop]=0;

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
      //      b[(grid->Nk[i]-1)-ktop]+=theta*dt*(bd[grid->Nk[i]-1]+2*alpha_bot*bd[grid->Nk[i]-1]);
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
	phys->wtmp2[i][k]+=sprop->wss[sdindx];
    }
    if(!(prop->TVD && prop->vertTVD))
      for(k=0;k<grid->Nk[i]+1;k++){
	
	ap[k] = 0.5*(phys->wtmp2[i][k]+fabs(phys->wtmp2[i][k]));
	am[k] = 0.5*(phys->wtmp2[i][k]-fabs(phys->wtmp2[i][k]));
       
      }
    else // Compute the ap/am for TVD schemes
      GetApAm(ap,am,phys->wp,phys->wm,phys->Cp,phys->Cm,phys->rp,phys->rm,
	      phys->wtmp2,grid->dzzold,phys->stmp,i,grid->Nk[i],ktop,prop->dt,prop->TVD);
    for(k=0;k<grid->Nk[i]+1;k++) {
	phys->wtmp2[i][k]-=sprop->wss[sdindx];
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

    //////////////Add sediment pickup/////////////////////////////
    if (sdindx == 0){
     
      if (grid->dv[i] > 5.0)       //tsungwei
	d[k-ktop] += dt*sedi->pickup[i]*0.1;
      else
	d[k-ktop] += dt*sedi->pickup[i]*sprop->btm_partition[sdindx];
    }else{
      //d[k-ktop] += dt*sedi->pickup[i];//*sprop->btm_partition[sdindx];;
      if (grid->dv[i] > 5.0)
	d[k-ktop] += dt*sedi->pickup[i]*0.9;
      else      
	d[k-ktop] += dt*sedi->pickup[i]*sprop->btm_partition[sdindx];
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

    // Now create the source term for the current time step
    for(k=0;k<grid->Nk[i];k++)
      ap[k]=0;
    
    if(!(prop->TVD && prop->horiTVD))
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
	
	if(prop->wetdry) {
	  for(k=0;k<grid->Nke[ne];k++)
	    ap[k] += dt*df*normal/Ac*
	      (theta*phys->u[ne][k]+(1-theta)*phys->utmp2[ne][k])*temp[k];
	} else {
	  for(k=0;k<grid->Nk[nc2];k++) 
	    ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]+fabs(phys->utmp2[ne][k]))*
	      sp[k]*grid->dzzold[nc2][k];
	  for(k=0;k<grid->Nk[nc1];k++) 
	    ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]-fabs(phys->utmp2[ne][k]))*
	      phys->stmp[nc1][k]*grid->dzzold[nc1][k];
	}	  
      }
    else {
      for(nf=0;nf<NFACES;nf++) {
	ne = grid->face[i*NFACES+nf];
	normal = grid->normal[i*NFACES+nf];
	df = grid->df[ne];
	dg = grid->dg[ne];
	nc1 = grid->grad[2*ne];
	nc2 = grid->grad[2*ne+1];
	if(nc1==-1) nc1=nc2;
	if(nc2==-1) nc2=nc1;
	if(nc2==-1) {
          if(boundary_scal && grid->mark[ne]==2)
            sp=phys->stmp2[nc1];
          else
            sp=phys->stmp[nc1];
        } else
          sp=phys->stmp[nc2];

	for(k=0;k<grid->Nk[nc2];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(theta*(phys->u[ne][k]+fabs(phys->u[ne][k]))+(1-theta)*(phys->utmp2[ne][k]+fabs(phys->utmp2[ne][k])))*sp[k]*grid->dzzold[nc2][k];
	// +0.5*dt/Ac*phys->SfHp[ne][k];
	for(k=0;k<grid->Nk[nc1];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(theta*(phys->u[ne][k]-fabs(phys->u[ne][k]))+(1-theta)*(phys->utmp2[ne][k]-fabs(phys->utmp2[ne][k])))*phys->stmp[nc1][k]*grid->dzzold[nc1][k];
	// +0.5*dt/Ac*phys->SfHm[ne][k];
	    
      }
      
      // wet/dry tvd
      for(k=0;k<grid->Nk[i];k++) 	
	if(grid->ctop[i]!=grid->Nk[i]-1)
	  ap[k] += 0.5*dt/Ac*phys->tvdminus[i][k] + 0.5*dt/Ac*phys->tvdplus[i][k];
    }

    for(k=ktop+1;k<grid->Nk[i];k++) 
      Cn[i][k-ktop]-=ap[k];
    
    for(k=0;k<=ktop;k++) 
      Cn[i][0]-=ap[k];

    for(k=0;k<grid->Nk[i];k++)
      sedi->G[i][k] = -ap[k];

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
      TriSolve(a,b,c,d,&(sd[i][ktop]),grid->Nk[i]-ktop);
    else if(b[0]!=0)
      sd[i][ktop]=d[0]/b[0];

    for(k=0;k<grid->ctop[i];k++)
      sd[i][k]=0;
    
    for(k=grid->ctop[i]; k<grid->ctopold[i];k++)
      sd[i][k] = sd[i][ktop];

    }
  
  if (sprop->Nsize != 1)
    if (sprop->size_exchange != 0)
      SizeExchange(grid, sedi, sprop, prop, sdindx, comm, myproc);
   
}


void ObtainSettlingVelocity(gridT *grid, physT *phys, sediT *sedi, spropT *sprop)
{
 int i, k, Nc=grid->Nc, NL=sprop->NL;
  REAL floc, hinder, gell, wsn, w0, tmp;
  
  w0 = -(sprop->spwght-1)*GRAV*pow(sprop->diam, 2)/(18*0.000001);
  floc = 10;
  hinder = 10000;
  gell = 180000;
  wsn = 2.0;

  for(i=0;i<Nc;i++){
    for(k=0;k<grid->Nk[i];k++){
      //      if (sedi->sd[i][k] < floc)
      //	sedi->ws[i][k] = w0;
      //      else if (sedi->sd[i][k] >= floc && sedi->sd[i][k] < hinder)
      //	sedi->ws[i][k] = w0*sedi->sd[i][k]/sprop->spwght;
//      else if (sedi->sd[i][k] >= hinder && sedi->sd[i][k] < gell){
//	tmp = (1-sedi->sd[i][k]/gell);
//	sedi->ws[i][k] = w0*hinder/sprop->spwght*pow(tmp,wsn);
//	  }
//      else{
//	    sedi->ws[i][k] = 0;
//	    sedi->sd[i][k] = gell;
//	  }
      sedi->ws[i][k] = -0.002;
    }
    sedi->ws[i][grid->Nk[i]] = sedi->ws[i][grid->Nk[i]-1];
  }
    
  

}

void CalculateBedSediment(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave)
{
  int i, kk, Nc=grid->Nc, NL=sprop->NL, indxBeforeErosion, indxAfterErosion, m;
  REAL z0tot, Cd, taub, angle, Tstar, tmp, cph, net_erosion, nextErosion;
  REAL pickup0, pickup0_tot;
  REAL alpha, a1, a2, ar, dt = prop->dt;
  REAL eta, eta_ano, lambda, d0, d0_eta_ano, d0_eta, d0_lambda, lambda_ano, lambda_orb;
  REAL A1, A2, A3, B1, B2, B3, C0, F;
  REAL db, dbbtm, dbtop, ub, vb, zbtm;

  alpha = 0.056;
  C0 = 1200000;
  a1 = 0.068;
  a2 = 0.0204*pow(log(sprop->diam),2)+0.022*log(sprop->diam)+0.0709;
  ar = 0.267;
  
  A1 = 0.095;
  A2 = 0.442;
  A3 = 2.28;
  
  B1 = 1/A1;
  B2 = 0.5*(1+A2)*B1;
  B3 = pow(B2, 2)-A3*B1;

  lambda_ano = 535*sprop->diam;
  eta_ano = 0.171*lambda_ano;


  for(i=0;i<Nc;i++){
    if (prop->wave){
      if(wave->ab[i] <= 0.2*sedi->kb[i])
	wave->fw[i] = 0.3;
      else if(wave->ab[i] <= 100*sedi->kb[i])
	wave->fw[i] = exp(-8.82+7.02*pow((wave->ab[i]/sedi->kb[i]), -0.140));
      else
	wave->fw[i] = exp(-7.30+5.61*pow((wave->ab[i]/sedi->kb[i]), -0.209));
      if(wave->fw[i] > 0.3) wave->fw[i] = 0.3;
    sedi->tau_w[i] = 0.5*wave->fw[i]*1000*pow(wave->ub[i],2);
    }
    z0tot = sedi->kb[i]/30;

    //In order to have consistent and stable sediment pickup, shear stress must
    //be obtained at the same level through the whole domain so that unphysically
    //high bottom stress won't be encountered when the level is too small. For
    //simplicity, I choose the level, zbtm, to be the smallest z-level through
    //the domain, which is 0.13 in the SF Bay case, so that it is guarantteed
    //that only one more point above the bottom-most cell is needed for
    //interpolation.          
    
    zbtm = 0.13;
    db = 0.5*(grid->dzz[i][grid->Nk[i]-1]+grid->dzz[i][grid->Nk[i]-2]);
    if(0.5*grid->dzz[i][grid->Nk[i]-1] < zbtm){ //interpolation

      dbbtm = (zbtm-0.5*grid->dzz[i][grid->Nk[i]-1])/db;
      dbtop = 1.0-dbbtm;

      ub = dbbtm*phys->uc[i][grid->Nk[i]-2]
	 + dbtop*phys->uc[i][grid->Nk[i]-1];
      vb = dbbtm*phys->vc[i][grid->Nk[i]-2]
	 + dbtop*phys->vc[i][grid->Nk[i]-1];
    }else{//extrapolation
      ub = phys->uc[i][grid->Nk[i]-1]-zbtm/db*(phys->uc[i][grid->Nk[i]-2]
					       -phys->uc[i][grid->Nk[i]-1]);
      vb = phys->vc[i][grid->Nk[i]-1]-zbtm/db*(phys->vc[i][grid->Nk[i]-2]
					       -phys->vc[i][grid->Nk[i]-1]);
    }


    Cd = 1/KAPPA_VK*log(zbtm/z0tot);
    Cd = pow(Cd, -2);
    //    sedi->tau_c[i] = Cd*1000*(pow(phys->uc[i][grid->Nk[i]-1], 2)
    //				  +  pow(phys->vc[i][grid->Nk[i]-1], 2));
    sedi->tau_c[i] = Cd*1000*(pow(ub, 2) + pow(vb, 2));
    
    if (phys->uc[i][grid->Nk[i]-1] != 0) 
      angle = atan(phys->vc[i][grid->Nk[i]-1]/phys->uc[i][grid->Nk[i]-1]);
    else
      if(phys->vc[i][grid->Nk[i]-1] > 0)
	angle = PI/2;
      else
	angle = 3*PI/2;
    if (prop->wave){
      wave->Cr[i] = angle-wave->thtamean[i];    
      taub = sqrt(pow(sedi->tau_c[i], 2)+pow(sedi->tau_w[i], 2)
		  +2*sedi->tau_c[i]*sedi->tau_w[i]*cos(wave->Cr[i]));
    }else{
      taub = sqrt(pow(sedi->tau_c[i], 2));
    }
    sedi->tau_c[i] = taub;
    //taub = sedi->tau_c[i];
    
    indxBeforeErosion = 0;
    indxAfterErosion = 0;
    while (sedi->M[i][indxBeforeErosion] <= 0 && indxBeforeErosion < sprop->NL-1){
      indxBeforeErosion += 1;
    }
    indxAfterErosion = indxBeforeErosion;
    
        
    //	Obtain the ripple field
    //   if (sedi->tau_w[i] > sedi->tau_ce[i][indx]){
    //  d0 = 2*wave->ab[i];
    // lambda_orb = 0.62*d0;
    //if (d0/eta_ano < 20)
    //lambda = 0.62*d0;
    //else if (d0/eta_ano <= 100)
    //lambda = lambda_ano*exp(-log(lambda_orb/lambda_ano)*log(0.01*d0/eta_ano)/log(5));
    //else
    //lambda = lambda_ano;
      
    //F = B3-B1*log(d0/lambda);
      
    //if (F < 0)
    //eta = 0;
    //else
    //eta = d0/(exp(B2-sqrt(F)));
    //sedi->z0r[i] = ar*pow(eta, 2)/lambda;
    //}
    //Finish paramerization of the ripple field
    
    //Obtain sediment pickup
    
    //if (indx == 0){
      //pickup of macro flocs
      //if (taub > sedi->tau_ce[i][indx]){
    //	pickup0 = sedi->E[indx]*exp(sedi->alpha[indx]*sqrt(taub-sedi->tau_ce[i][indx]));
    //}else{
    //pickup0 = 0;  
    //}
      
      //pickup of micro flocs
      //      if (taub > sedi->tau_ce_fine){
      //pickup0_fine = sedi->E_fine*exp(sedi->alpha_fine*sqrt(taub-sedi->tau_ce_fine));
      //if (pickup0*dt > sedi->M[i][indx]) pickup0_fine = sedi->M[i][indx]/dt;
      //}else{
      //pickup0_fine = 0;  
	//}
      //total pickup = macro + micro
      //pickup0_tot = pickup0;

      //if total pickup > mass in the first bed layer, total pick = mass in the first layer
      //and the pickup for each size class is redistributed according to the proportion of
      //its pickup magnitude to the total erosion.
      //      if (pickup0_tot*dt > sedi->M[i][indx]){	
      //	pickup0_tot = sedi->M[i][indx]/dt;
      //	pickup0 = sedi->M[i][indx]/dt*pickup0/pickup0_tot;
      	//pickup0_fine = sedi->M[i][indx]/dt*pickup0_fine/pickup0_tot; 
      //    }else{
      //	pickup0 = 0;
      //	pickup0_fine = 0;
      //pickup0_tot = 0;
      //}    
      //}
    Tstar = taub/sedi->tau_ce[i][indxBeforeErosion];
    if (Tstar > 1.0){
      sedi->z0b[i] = alpha*sprop->diam*a1*Tstar/(1+a2*Tstar);
      pickup0 = sedi->E[indxBeforeErosion]*exp(sedi->alpha[indxBeforeErosion]*sqrt(taub-sedi->tau_ce[i][indxBeforeErosion]));
    }else
      pickup0 = 0.0;
    

    
    if (pickup0 != pickup0) printf("-----------------------------pickup = %f\n", pickup0);
    
    net_erosion = pickup0*dt + sedi->ws[i][grid->Nk[i]]*dt; 
    sedi->dpsit[i] = sedi->dpsit[i]-net_erosion;
    if(net_erosion <= 0)
      sedi->M[i][indxBeforeErosion] -= net_erosion; //new deposition always becomes the first soft mud layer
    else{
      sedi->M[i][indxBeforeErosion] -= net_erosion;
      nextErosion = net_erosion-sedi->M[i][indxBeforeErosion];
      while (sedi->M[i][indxAfterErosion] <= 0.0 && indxAfterErosion < sprop->NL-1){
	sedi->M[i][indxAfterErosion] = 0.0;
	nextErosion = nextErosion - sedi->M[i][indxAfterErosion];
	indxAfterErosion = indxAfterErosion + 1;
	sedi->M[i][indxAfterErosion] = sedi->M[i][indxAfterErosion] - nextErosion;	  
      }      
    }
    if (indxAfterErosion > sprop->NL-1) indxAfterErosion = sprop->NL-1;
    C0=sedi->Dl[indxAfterErosion];
    sedi->pickup[i] = pickup0*(1-sedi->sdtot[i][grid->Nk[i]-1]/C0)/(1+pickup0*dt/grid->dzz[i][grid->Nk[i]-1]/C0);

    if (prop->wave){
      if (wave->ub[i] > 2){
	cph = 100000;
	for (m = 0; m < 10; m++){
	  cph = Min(cph, wave->cph[m][i]);
	}
      
	printf("hs = %f; ub = %f; sg=%f, k = %f, pickup = %f, taub = %f, depth = %f\n", wave->Hs[i], wave->ub[i], wave->sgmean[i], wave->kmean[i], tmp, taub, grid->dv[i]+phys->h[i]);
      }
    }


    //sedi->ws is the vertical sediment flux, which is always negative

    for (kk = indxAfterErosion; kk < sprop->NL-1; kk++){
      if (kk > 0){
	if (taub > sedi->tau_ce[i][kk]){
	  tmp = dt*sedi->E[kk]*exp(sedi->alpha[kk]*sqrt(taub-sedi->tau_ce[i][kk]));
	  if (kk == sprop->NL-2) tmp = tmp*sedi->bedmudratio[i];	  

	  //top of middle layers: - erosion at top 	    
	  sedi->M[i][kk] -= tmp;
	  //the upper layers: + erosion at top of the current middle layer 	    
	  sedi->M[i][kk-1] += tmp;
	}
      }
      if (kk > indxAfterErosion){ //no consolidation occurs at the top layer
	sedi->M[i][kk] += sedi->cnsd[kk-1]*dt;  //top of middle layers:  + consolidation from the above layer;       

      }
      //all layers: - current layer consolidation
      sedi->M[i][kk] -= sedi->cnsd[kk]*dt;
    }
    
    //sedi->kb[i] = 30*Max(sedi->z0s[i], sedi->z0b[i]+sedi->z0r[i]);
    //sedi->kb[i] = 30*sedi->z0s[i];	
  }
}


void CalculateTotalSediment(gridT *grid, sediT *sedi, spropT *sprop){
  int i, k, s, Nc = grid->Nc, Ns = sprop->Nsize;


  for(i = 0; i < Nc; i++){
    for(k = 0; k < grid->Nk[i]; k++){
      if(sedi->sd[0][i][k] < 0) sedi->sd[0][i][k] = 0;
      sedi->sdtot[i][k] = sedi->sd[0][i][k];
      sedi->ws[i][k] = sedi->sd[0][i][k]*sprop->wss[0];
      //Kurt Nelson added
      sedi->scc0[i][k] = sedi->sd[0][i][k]; //storing concentration of size class 0
      sedi->scc1[i][k] = sedi->sd[1][i][k]; //storing concentration of size class 1
      ///////////////////////////////////////////////////////////// 
    }
  }
  
  
  for(i = 0; i < Nc; i++){
    for(k = 0; k < grid->Nk[i]; k++){
      for(s = 1; s < Ns; s++){
	if(sedi->sd[s][i][k] < 0) sedi->sd[s][i][k] = 0;
	sedi->sdtot[i][k] += sedi->sd[s][i][k];
	sedi->ws[i][k] += sedi->sd[s][i][k]*sprop->wss[s];
      }      
    }
    if(grid->Nk[i] > 2)
      sedi->ws[i][grid->Nk[i]] = 2.0*sedi->ws[i][grid->Nk[i]-1]-sedi->ws[i][grid->Nk[i]-2];
    else
      sedi->ws[i][grid->Nk[i]] = sedi->ws[i][grid->Nk[i]-1];
  }
  

  //  for(i = 0; i < Nc; i++){
  //    for(k = 0; k < grid->Nk[i]; k++){
  //      if(sedi->sdtot[i][k] < SMALL)
  //      	sedi->ws[i][k] = 0;
  //      else
  //  	sedi->ws[i][k] /= sedi->sdtot[i][k];
  //    }
  //  }

}

void SizeExchange(gridT *grid, sediT *sedi, spropT *sprop, propT *prop, int sdindx, MPI_Comm comm, int myproc){
  int i, k, s=sdindx, Nc = grid->Nc, Ns = sprop->Nsize;
  REAL ka=sprop->Kagg, kb=sprop->Kbrk, q=sprop->q, p=sprop->p;
  REAL dt=prop->dt, dD, fct, cfl; 
  REAL sdold[Ns];
 
  if (sdindx == 1){ //This is only for 2-class model. In the future, this definitely needs to be changed.
    for(i = 0; i < Nc; i++){
      for(k = 0; k < grid->Nk[i]; k++){
	sedi->aggreg[s][i][k] = ka*sedi->sd[s-1][i][k]*sedi->G[i][k]
	  *pow(sprop->diam_repr[s-1], 4.0-sprop->nf);
	sedi->breakup[s][i][k] = kb*pow(sedi->G[i][k], q+1.0)*pow(sprop->diam_repr[s], 2.0*q+1)
	  * pow(sprop->diam_repr[s]-sprop->diam_repr[s-1], p);
	dD = sprop->diam_repr[s] - sprop->diam_repr[s-1];
	cfl = (sedi->aggreg[s][i][k]-sedi->breakup[s][i][k])*dt/dD;
	//	if (cfl >= 1.0)
	//	  printf("Size exchange too fast; cfl = %f\n", cfl);
	sdold[s] = sedi->sd[s][i][k];
	sdold[s-1] = sedi->sd[s-1][i][k];
	sedi->sd[s][i][k] = sedi->sd[s][i][k] + 0.5*(cfl+fabs(cfl))*sedi->sd[s-1][i][k]
	  + 0.5*(cfl-fabs(cfl))*sedi->sd[s][i][k];
	sedi->sd[s-1][i][k] = sedi->sd[s-1][i][k] - 0.5*(cfl+fabs(cfl))*sedi->sd[s-1][i][k]
	  - 0.5*(cfl-fabs(cfl))*sedi->sd[s][i][k];
	if (sedi->sd[s][i][k] < 0.0){
	  sedi->sd[s][i][k] = 0.0;
	  sedi->sd[s-1][i][k] = sdold[s] + sdold[s-1];
	}else if(sedi->sd[s-1][i][k] < 0.0){
	  sedi->sd[s-1][i][k] = 0.0;
	  sedi->sd[s][i][k] = sdold[s] + sdold[s-1];
	}
	  
      }
    }
  }
}


void ImplicitSizeExchange(gridT *grid, sediT *sedi, spropT *sprop, propT *prop, MPI_Comm comm, int myproc){
  int i, k, s, Nc = grid->Nc, Ns = sprop->Nsize;
  REAL ka=sprop->Kagg, kb=sprop->Kbrk, q=sprop->q, p=sprop->p;
  REAL dt=prop->dt, dD, fct, cfl; 
  REAL x[Ns], a[Ns], b[Ns], c[Ns], d[Ns];

  for(s = 1; s < Ns; s++){  
    for(i = 0; i < Nc; i++){
      for(k = 0; k < grid->Nk[i]; k++){
	sedi->aggreg[s][i][k] = ka*sedi->sd[s-1][i][k]*sedi->G[i][k]
	  *pow(sprop->diam_repr[s-1], 4.0-sprop->nf);
	sedi->breakup[s][i][k] = kb*pow(sedi->G[i][k], q+1.0)*pow(sprop->diam_repr[s], 2.0*q+1)
	  * pow(sprop->diam_repr[s]-sprop->diam_repr[s-1], p);
	cfl = fabs(sedi->aggreg[s][i][k]-sedi->breakup[s][i][k])*dt/dD;
	//	if (cfl >= 1.0)
	//	  printf("Size exchange too fast; cfl = %f\n", cfl);	  
      }
    }
  }
  
  for(i = 0; i < Nc; i++){
    for(k = 0; k < grid->Nk[i]; k++){
      for(s = 0; s < Ns; s++){

 	if (s == 0)
	  dD = sprop->diam_repr[s+1] - sprop->diam_repr[s];
	else
 	  dD = sprop->diam_repr[s] - sprop->diam_repr[s-1];

 	fct = dt/(2.0*dD);
	 
	if (s == Ns-1){
	  a[s] = fct*sedi->aggreg[s][i][k];
	  b[s] = 1 + fct*(-sedi->breakup[s][i][k]);
	  c[s] = 0;;
	}else{
	  a[s] = fct*sedi->aggreg[s][i][k];
	  b[s] = 1 + fct*(-sedi->breakup[s][i][k]-sedi->aggreg[s+1][i][k]);
	  c[s] = fct*sedi->breakup[s+1][i][k];
	}
	
	if (s == 0)
	  d[s] = b[s]*sedi->sd[s][i][k] + c[s]*sedi->sd[s+1][i][k];
	else if (s == Ns-1)
	  d[s] = a[s]*sedi->sd[s-1][i][k] + b[s]*sedi->sd[s][i][k];
	else
	  d[s] = a[s]*sedi->sd[s-1][i][k] + b[s]*sedi->sd[s][i][k]
	    + c[s]*sedi->sd[s+1][i][k];
      }

      for(s = 0; s < Ns; s++){

	if (s == 0)
	  dD = sprop->diam_repr[s+1] - sprop->diam_repr[s];
	else
	  dD = sprop->diam_repr[s] - sprop->diam_repr[s-1];
	
	fct = dt/(2.0*dD);
	
	if (s == Ns-1){
	  a[s] = -fct*sedi->aggreg[s][i][k];
	  b[s] = 1 - fct*(-sedi->breakup[s][i][k]);
	  c[s] = 0;
	}else{
	  a[s] = -fct*sedi->aggreg[s][i][k];
	  b[s] = 1 - fct*(-sedi->breakup[s][i][k]-sedi->aggreg[s+1][i][k]);
	  c[s] = -fct*sedi->breakup[s+1][i][k];
	}
	x[s] = 0;
      }
      TriSolve(a, b, c, d, (&x[0]), Ns);
      
      for (s=0; s < Ns; s++)
	sedi->sd[s][i][k] = x[s];
    }
  }

}



void UpdateSedi_NoHorizTransport(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave, REAL **sd, int sdindx,
                REAL **boundary_scal, REAL **Cn, REAL kappa, REAL kappaH, REAL **kappa_tv, REAL theta,
		REAL **src1, REAL **src2, REAL *Ftop, REAL *Fbot, int alpha_top, int alpha_bot,
		MPI_Comm comm, int myproc)
{
  int i, iptr, j, jptr, ib, k, nf, ktop, indx, kk, indxx;
  int Nc=grid->Nc, normal, nc1, nc2, ne;
  REAL df, dg, Ac, dt=prop->dt, fab, *a, *b, *c, *d, *ap, *am, *bd, dznew, mass, *sp, *temp,
    Cdavg, U_sq, taub, cb, wsc, Uf, pec, tau_e, angle, C0;


  prop->TVD = TVDMACRO;
  // These are used mostly debugging to turn on/off vertical and horizontal TVD.
  prop->horiTVD = 1;
  prop->vertTVD = 1;
  C0 = sedi->Dl[0];

  ap = phys->ap;
  am = phys->am;
  bd = phys->bp;
  temp = phys->bm;
  a = phys->a;
  b = phys->b;
  c = phys->c;
  d = phys->d;
  
  //ObtainSettlingVelocity(grid, phys, sedi, sprop);     
  // Don't use AB2 when wetting and drying is employed or if
  // using the TVD schemes.
  if(prop->n==1 || prop->wetdry || prop->TVD!=0) {
    fab=1;
    for(i=0;i<grid->Nc;i++)
      for(k=0;k<grid->Nk[i];k++)
	Cn[i][k]=0;
  } else
    fab=1.5;

    for(i=0;i<Nc;i++) 
      for(k=0;k<grid->Nk[i];k++) 
	phys->stmp[i][k]=sd[i][k];

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
  
  // Compute the scalar on the vertical faces (for horiz. advection)
  //if(prop->TVD && prop->horiTVD)
  //  HorizontalFaceScalars(grid,phys,prop,boundary_scal,prop->TVD,comm,myproc); 

  for(iptr=grid->celldist[0];iptr<grid->celldist[1];iptr++){
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

    for(k=0;k<grid->Nk[i]+1;k++) {
      phys->w[i][k]+=sprop->wss[sdindx];
    }
    // These are the advective components of the tridiagonal
    // at the new time step.
    if(!(prop->TVD && prop->vertTVD))
      
      for(k=0;k<grid->Nk[i]+1;k++) {
	ap[k] = 0.5*(phys->w[i][k]+fabs(phys->w[i][k]));
	am[k] = 0.5*(phys->w[i][k]-fabs(phys->w[i][k]));
	}
    else  // Compute the ap/am for TVD schemes
      GetApAm(ap,am,phys->wp,phys->wm,phys->Cp,phys->Cm,phys->rp,phys->rm,
	      phys->w,grid->dzz,sd,i,grid->Nk[i],ktop,prop->dt,prop->TVD);
      
    for(k=0;k<grid->Nk[i]+1;k++) {
      phys->w[i][k]-=sprop->wss[sdindx];
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
    //   b[(grid->Nk[i]-1)-ktop]+=c[(grid->Nk[i]-1)-ktop];
    // Change for sediment BC. 
    // Since the flux through the bottom boundary is represented by pickup,
    // no vertical velocity at the bottom boundary is considered in the solver. 

      //      b[(grid->Nk[i]-1)-ktop]+=theta*dt*am[grid->Nk[i]-ktop];
      //      c[(grid->Nk[i]-1)-ktop]=0;

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
      //      b[(grid->Nk[i]-1)-ktop]+=theta*dt*(bd[grid->Nk[i]-1]+2*alpha_bot*bd[grid->Nk[i]-1]);
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
	phys->wtmp2[i][k]+=sprop->wss[sdindx];
    }
    if(!(prop->TVD && prop->vertTVD))
      for(k=0;k<grid->Nk[i]+1;k++){
	
	ap[k] = 0.5*(phys->wtmp2[i][k]+fabs(phys->wtmp2[i][k]));
	am[k] = 0.5*(phys->wtmp2[i][k]-fabs(phys->wtmp2[i][k]));
       
      }
    else // Compute the ap/am for TVD schemes
      GetApAm(ap,am,phys->wp,phys->wm,phys->Cp,phys->Cm,phys->rp,phys->rm,
	      phys->wtmp2,grid->dzzold,phys->stmp,i,grid->Nk[i],ktop,prop->dt,prop->TVD);
    for(k=0;k<grid->Nk[i]+1;k++) {
	phys->wtmp2[i][k]-=sprop->wss[sdindx];
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

    //////////////Add sediment pickup/////////////////////////////
    if (sdindx == 0){
      d[k-ktop] += dt*sedi->pickup[i]*sprop->btm_partition[sdindx];
    }else{
      //d[k-ktop] += dt*sedi->pickup[i];//*sprop->btm_partition[sdindx];;
      d[k-ktop] += dt*sedi->pickup[i]*sprop->btm_partition[sdindx];
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

    // Now create the source term for the current time step
    // Below is deleted for the case of no horizontal transport
    /*    for(k=0;k<grid->Nk[i];k++)
      ap[k]=0;
    
    if(!(prop->TVD && prop->horiTVD))
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
	
	if(prop->wetdry) {
	  for(k=0;k<grid->Nke[ne];k++)
	    ap[k] += dt*df*normal/Ac*
	      (theta*phys->u[ne][k]+(1-theta)*phys->utmp2[ne][k])*temp[k];
	} else {
	  for(k=0;k<grid->Nk[nc2];k++) 
	    ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]+fabs(phys->utmp2[ne][k]))*
	      sp[k]*grid->dzzold[nc2][k];
	  for(k=0;k<grid->Nk[nc1];k++) 
	    ap[k] += 0.5*dt*df*normal/Ac*(phys->utmp2[ne][k]-fabs(phys->utmp2[ne][k]))*
	      phys->stmp[nc1][k]*grid->dzzold[nc1][k];
	}	  
      }
    else {
      for(nf=0;nf<NFACES;nf++) {
	ne = grid->face[i*NFACES+nf];
	normal = grid->normal[i*NFACES+nf];
	df = grid->df[ne];
	dg = grid->dg[ne];
	nc1 = grid->grad[2*ne];
	nc2 = grid->grad[2*ne+1];
	if(nc1==-1) nc1=nc2;
	if(nc2==-1) nc2=nc1;
	if(nc2==-1) {
          if(boundary_scal && grid->mark[ne]==2)
            sp=phys->stmp2[nc1];
          else
            sp=phys->stmp[nc1];
        } else
          sp=phys->stmp[nc2];

	for(k=0;k<grid->Nk[nc2];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(theta*(phys->u[ne][k]+fabs(phys->u[ne][k]))+(1-theta)*(phys->utmp2[ne][k]+fabs(phys->utmp2[ne][k])))*sp[k]*grid->dzzold[nc2][k];
	// +0.5*dt/Ac*phys->SfHp[ne][k];
	for(k=0;k<grid->Nk[nc1];k++) 
	  ap[k] += 0.5*dt*df*normal/Ac*(theta*(phys->u[ne][k]-fabs(phys->u[ne][k]))+(1-theta)*(phys->utmp2[ne][k]-fabs(phys->utmp2[ne][k])))*phys->stmp[nc1][k]*grid->dzzold[nc1][k];
	// +0.5*dt/Ac*phys->SfHm[ne][k];
	    
      }
      
      // wet/dry tvd
      for(k=0;k<grid->Nk[i];k++) 	
	if(grid->ctop[i]!=grid->Nk[i]-1)
	  ap[k] += 0.5*dt/Ac*phys->tvdminus[i][k] + 0.5*dt/Ac*phys->tvdplus[i][k];
    }

      for(k=ktop+1;k<grid->Nk[i];k++) 
      Cn[i][k-ktop]-=ap[k];
    
    for(k=0;k<=ktop;k++) 
      Cn[i][0]-=ap[k];

    // Add on the source from the current time step to the rhs.
    for(k=0;k<grid->Nk[i]-ktop;k++) 
    d[k]+=fab*Cn[i][k];*/ 

    for(k=ktop;k<grid->Nk[i];k++)
      ap[k]=Cn[i][k-ktop];
    for(k=0;k<=ktop;k++)
      Cn[i][k]=0;
    for(k=ktop+1;k<grid->Nk[i];k++)
      Cn[i][k]=ap[k];
    for(k=grid->ctop[i];k<=ktop;k++)
      Cn[i][k]=ap[ktop]/(1+fabs(grid->ctop[i]-ktop));

    if(grid->Nk[i]-ktop>1) 
      TriSolve(a,b,c,d,&(sd[i][ktop]),grid->Nk[i]-ktop);
    else if(b[0]!=0)
      sd[i][ktop]=d[0]/b[0];

    for(k=0;k<grid->ctop[i];k++)
      sd[i][k]=0;
    
    for(k=grid->ctop[i]; k<grid->ctopold[i];k++)
      sd[i][k] = sd[i][ktop];

    }
  
  if (sprop->Nsize != 1)
    if (sprop->size_exchange != 0)
      SizeExchange(grid, sedi, sprop, prop, sdindx, comm, myproc);
   
}
