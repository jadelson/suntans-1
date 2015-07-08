# Makefile for SUNTANS
#
# Oliver Fringer
# Stanford University
# fringer@stanford.edu
#
# Need to define appropriate directories in Makefile.in first!
#
# Copyright (C) 2005-2006 The Board of Trustees of the Leland Stanford Junior University. All Rights Reserved.
#                  
include Makefile.in

CC = $(MPIHOME)/bin/mpicc -std=c99 -w

XLDFLAGS=-lX11 -lm
XINC=/usr/include
XLIBDIR = /usr/lib64

#XINC=/usr/X11R6/include
#XLIBDIR = /usr/X11R6/lib64

ifneq ($(TRIANGLEHOME),)
  TRIANGLEINCLUDE = -I$(TRIANGLEHOME)
  TRIANGLELIB = $(TRIANGLEHOME)/triangle.o
  TRIANGLELIBDIR = -L$(TRIANGLEHOME)
  TRIANGLELD = 
  TRIANGLESRC = triangulate.c
else
  TRIANGLEINCLUDE =
  TRIANGLELIB =
  TRIANGLELIBDIR =
  TRIANGLELD =
  TRIANGLESRC = triangulate-notriangle.c
endif	

ifneq ($(PARMETISHOME),)
  PARMETISINCLUDE = -I$(PARMETISHOME)/ParMETISLib
  PARMETISLIB = $(PARMETISHOME)/libparmetis.a $(PARMETISHOME)/libmetis.a
  PARMETISLIBDIR = -L$(PARMETISHOME)
  PARMETISLD = -lparmetis -lmetis 
  PARMETISSRC = partition.c
else
  PARMETISINCLUDE =
  PARMETISLIB =
  PARMETISLIBDIR =
  PARMETISLD =
  PARMETISSRC = partition-noparmetis.c
endif

LD = $(CC) 
LIBS = $(PARMETISLIB) $(TRIANGLELIB)
LIBDIR = $(PARMETISLIBDIR) $(TRIANGLELIBDIR)
LDFLAGS = -lm $(LIBDIR) $(LIBS)
INCLUDES = $(PARMETISINCLUDE) $(TRIANGLEINCLUDE)
CFLAGS = $(OPTFLAGS) $(INCLUDES)
MATHLIB = -lm

EXEC = sun
PEXEC = sunplot

DEPFLAGS = -Y

SRCS =  mympi.c grid.c report.c util.c fileio.c phys.c suntans.c initialization.c memory.c timer.c state.c sedi.c\
	turbulence.c boundaries.c profiles.c scalars.c tvd.c wave.c\
	$(TRIANGLESRC) 
OBJS = $(SRCS:.c=.o)

PLOTSRCS = sunplot.c fileio.c
PLOTOBJS = $(PLOTSRCS:.c=.o)

all:	$(EXEC)

.c.o:	
	$(CC) $(CFLAGS) -c $*.c

$(EXEC): $(OBJS) 
	$(LD)  -o $@ $(OBJS) $(LDFLAGS)

$(PEXEC): $(PLOTOBJS)
	$(LD) -o sunplot $(PLOTOBJS) $(XLDFLAGS) -L$(XLIBDIR)

kriging.o:	kriging.c
	$(LD) -c kriging.c $(MATHLIB)

depend: 
	makedepend $(DEPFLAGS) -- $(SRCS) $(PLOTSRCS) &> /dev/null

clean:
	rm -f *.o

clobber:	clean
	rm -f *~ \#*\# PI* $(EXEC) $(PEXEC) $(DEPFILE)
	make -C examples clobber

# DO NOT DELETE THIS LINE - Dependencies are appended after it.

mympi.o: mympi.h suntans.h fileio.h
grid.o: grid.h suntans.h fileio.h mympi.h util.h initialization.h memory.h
grid.o: triangulate.h report.h
report.o: report.h grid.h suntans.h fileio.h mympi.h
util.o: grid.h suntans.h fileio.h mympi.h util.h
fileio.o: fileio.h
phys.o: suntans.h phys.h grid.h fileio.h mympi.h util.h initialization.h
phys.o: memory.h turbulence.h boundaries.h profiles.h state.h scalars.h
phys.o: sedi.h wave.h
suntans.o: suntans.h mympi.h fileio.h grid.h phys.h sedi.h report.h wave.h
initialization.o: fileio.h suntans.h initialization.h
memory.o: memory.h
timer.o: mympi.h suntans.h fileio.h timer.h
state.o: state.h grid.h suntans.h fileio.h mympi.h phys.h
sedi.o: sedi.h suntans.h grid.h fileio.h mympi.h phys.h util.h tvd.h memory.h
sedi.o: turbulence.h
turbulence.o: phys.h suntans.h grid.h fileio.h mympi.h util.h turbulence.h
turbulence.o: boundaries.h scalars.h
boundaries.o: boundaries.h mympi.h suntans.h fileio.h phys.h grid.h
profiles.o: util.h grid.h suntans.h fileio.h mympi.h memory.h phys.h
profiles.o: profiles.h
scalars.o: scalars.h suntans.h grid.h fileio.h mympi.h phys.h util.h tvd.h
tvd.o: suntans.h phys.h grid.h fileio.h mympi.h tvd.h util.h
wave.o: grid.h suntans.h fileio.h mympi.h wave.h phys.h util.h tvd.h memory.h
triangulate.o: suntans.h mympi.h fileio.h grid.h triangulate.h
sunplot.o: suntans.h fileio.h
fileio.o: fileio.h
