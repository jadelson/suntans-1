/*
 * File: elm.h
 * Author: Oliver B. Fringer
 * Institution: Stanford University
 * --------------------------------
 * Header file for elm.c
 *
 * Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior
 * University. All Rights Reserved.
 *
 */
#ifndef _elm_h
#define _elm_h

#include "phys.h"
#include "suntans.h"
#include "grid.h"
#include "fileio.h"
                                                                             
void LagraTracing(gridT *grid, physT *phys, propT *prop, int start_cell, int start_layer, int *ii, int *kk, REAL *xs, REAL *zs, int myproc);
void InterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us, REAL *al, REAL *vtx, REAL *vty,int ifstdout);
void StableInterpVelo(gridT *grid, physT *phys, int i, int k, REAL *xs,REAL *us);

#endif
