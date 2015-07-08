/*
 * File: elm.c
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * This file contains elm functions.
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior
 * University. All Rights Reserved. 
 *
 */
#include <stdio.h>
#include <string.h>
#include "suntans.h"
#include "phys.h"
#include "grid.h"
#include "util.h"
#include "memory.h"
#include "profiles.h"
#include "elm.h"

#define TRUE 1
#define FALSE 0


/*
 * Function: LagraTracing
 * Usage:
 * -------------------------------------------------
 * Tracing back from a given point, following the Lagrangian trajectory.
 * ii,kk give the start cell number and vertical level
 * xs, zs are input as initial location, and updated to be the trace back location 
 * Horizontal tracing is not allowed to be more than 5 cells.
 *
 * elm version
 *
 */
void LagraTracing(gridT *grid, physT *phys, propT *prop, int start_cell, int start_layer, int *ii, int *kk, REAL *xs,REAL *zs,int myproc)
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
      dtx=Min(dtx,al[nf]);
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


    //Compute Horizontal Tracing
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

 if(iterx==5)  printf("Warning! Horizontal tracing crossed more than 2 cells. cell: %d, layer: %d \n", start_cell, start_layer);
 if(*ii==-1)*ii = i2;
 sync();
}

/*
 * elm version - function
 *
 */
void InterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us, REAL *al, REAL *vtx, REAL *vty,int debugvelo)
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

/*
 * elm version - function
 *
 */
void StableInterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us)
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
