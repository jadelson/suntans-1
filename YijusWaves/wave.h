/*
 *File: wave.h
 *Author: Yi-Ju Chou
 *
 *
 */

#ifndef _wave_h
#define _wave_h

#include "suntans.h"
#include "grid.h"
#include "phys.h"

void InitializeWaveProperties(wpropT **wprop, propT *prop, int myproc);
void AllocateWaveVariables(gridT *grid, waveT **wave, propT *prop, wpropT *wprop);
void FreeWaveVariables(gridT *grid, waveT *wave, propT *prop, wpropT *wprop);
void InitializeWaveVariables(gridT *grid, waveT *wave, propT *prop, wpropT *wprop, int myproc, MPI_Comm comm);
void InputWind(int statio, propT *prop, waveT *wave, wpropT *wprop, int myproc, int numprocs);
void ObtainKrigingCoef(gridT *grid, waveT *wave, wpropT *wprop, int myproc, int numprocs);
void WindField(propT *prop, gridT *grid, physT *phys, waveT *wave, wpropT *wprop, int myproc, int numprocs);
void UpdateWave(gridT *grid, physT *phys, waveT *wave, propT *prop, wpropT *wprop, sediT *sedi, spropT *sprop,
                MPI_Comm comm, int myproc, int numprocs);

#endif
