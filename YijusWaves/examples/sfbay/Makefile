include parameters.in
include $(SUNTANSHOME)/Makefile.in

SCRIPTHOME=../scripts/bobstar-pbs

CC = $(MPIHOME)/bin/mpicc -std=c99 -pedantic
LD = $(CC) 
CFLAGS = 
MATHLIB = -lm

OBJS = 
SUN = $(SUNTANSHOME)/sun
INCLUDES = -I$(SUNTANSHOME) -I$(MPIHOME) -I$(PARMETISHOME)/ParMETISLib

all:	data

script:	parameters.in
	sh ${SCRIPTHOME}/make-script.sh

test:	script data
	qsub $(SCRIPTFILE)

data:	$(SUN)

.c.o:	
	$(LD) -c $(INCLUDES) $*.c

$(SUN):	initialization.o boundaries.o
	cp initialization.o boundaries.o $(SUNTANSHOME)
	make -C $(SUNTANSHOME)
	cp $(SUNTANSHOME)/sun .

clean:
	rm -f *.o

clobber: clean
	rm -rf sun *~ \#*\# PI* $(EXEC) gmon.out data rundata/*~ $(TAGNAME).o* $(TAGNAME).e* $(SCRIPTFILE)

cleanrun:
	rm -rf $(SCRATCH)/$(TAGNAME)

newrun:
	rm -f $(SCRATCH)/$(TAGNAME)/{runnum.txt,crash.txt}

kill:
	qdel -u fringer

