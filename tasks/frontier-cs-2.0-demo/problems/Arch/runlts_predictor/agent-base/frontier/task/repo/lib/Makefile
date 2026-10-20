# /*
# 
# Copyright (c) 2019, North Carolina State University
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
# trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
# simulation thereof owned by North Carolina State University must not be used to endorse or promote products derived from this software without prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# */
# 
# // Author: Eric Rotenberg (ericro@ncsu.edu)


CC = g++
OPT = -O3
TOP = ..
INC = -I$(TOP) -I$(TOP)/lib
LIBS =
DEFINES = -DGZSTREAM_NAMESPACE=gz
FLAGS = -std=c++17 $(INC) $(LIBS) $(OPT) $(DEFINES)

ifeq ($(DEBUG), 1)
	CC += -ggdb3
endif

OBJ = cbp.o my_value_predictor.o parameters.o uarchsim.o cache.o bp.o resource_schedule.o gzstream.o
DEPS = $(TOP)/cbp.h value_predictor_interface.h sim_common_structs.h my_value_predictor.h trace_reader.h fifo.h parameters.h uarchsim.h cache.h bp.h resource_schedule.h gzstream.h

all: libcbp.a

libcbp.a: $(OBJ)
	ar r $@ $^

%.o: %.cc $(DEPS)
	$(CC) $(FLAGS) -c -o $@ $<


.PHONY: clean

clean:
	rm -f *.o libcbp.a
