/*
 *File: sedi.h
 *Author: Yi-Ju Chou
 *
 *
 */

#ifndef _sedi_h
#define _sedi_h

#include "suntans.h"
#include "grid.h"
#include "phys.h"

void InitializeSediProperties(propT *prop, spropT **sprop, int myproc);
void UpdateSedi(gridT *grid, physT *phys, propT *prop, sediT *sedi, 
                spropT *sprop, waveT *wave, REAL **sd, int sdindx, REAL **boundary_scal, REAL **Cn, 
		REAL kappa, REAL kappaH, REAL **kappa_tv, REAL theta,
		REAL **src1, REAL **src2, REAL *Ftop, REAL *Fbot, int alpha_top, int alpha_bot,
		MPI_Comm comm, int myproc);

void AllocateSediVariables(gridT *grid, sediT **sedi, spropT *sprop, propT *prop);
void FreeSediVariables(gridT *grid, sediT *sedi, propT *prop, spropT *sprop);
void InitializeSediVariables(gridT *grid, sediT *sedi, propT *prop, spropT *sprop, int myproc, MPI_Comm comm);
void ReadSediVariables(gridT *grid, sediT *sedi, propT *prop, int myproc, MPI_Comm comm);
void ObtainSettlingVelocity(gridT *grid, physT *phys, sediT *sedi, spropT *sprop);
void CalculateBedSediment(gridT *grid, physT *phys, propT *prop, sediT *sedi, spropT *sprop, waveT *wave);
void CalculateTotalSediment(gridT *grid, sediT *sedi, spropT *sprop);
//void UpdateOrbitalVel(sediT *sedi, spropT *sprop, gridT *grid, physT *phys, propT *prop, int myproc, int numprocs);
#endif
