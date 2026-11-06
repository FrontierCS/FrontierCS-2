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
LIBS = -lcbp -lz
#FLAGS = -std=c++11 -L./lib $(LIBS) $(OPT)
FLAGS = -std=c++17 -L./lib $(LIBS) $(OPT)
CPPFLAGS = -std=c++17 $(OPT)

OBJ = cond_branch_predictor_interface.o my_cond_branch_predictor.o
DEPS = cbp.h cond_branch_predictor_interface.h my_cond_branch_predictor.h

DEBUG=0
ifeq ($(DEBUG), 1)
	CC += -ggdb3
endif


.PHONY: clean lib

all: cbp

lib:
	make -C $@ DEBUG=$(DEBUG)

cbp: $(OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

%.o: %.cc $(DEPS)
	$(CC) $(FLAGS) -c -o $@ $<


clean:
	rm -f *.o cbp
	make -C lib clean
