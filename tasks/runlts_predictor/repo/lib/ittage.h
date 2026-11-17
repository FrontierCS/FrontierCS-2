/*Copyright (c) <2018>, INRIA : Institut National de Recherche en Informatique
et en Automatique (French National Research Institute for Computer Science and
Applied Mathematics) All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.*/

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#ifndef _ITTAGE_H
#define _ITTAGE_H

#define HISTBUFFERLENGTH                                                       \
  4096 // we use a 4K entries history buffer to store the branch history (this
       // allows us to explore using history length up to 4K)
#define BORNTICK 1024

#define ALTWIDTH 5

// Fast  implementation of the ITTAGE predictor: probably not optimal, but not
// that far
//
// utility class for index computation
// this is the cyclic shift register for folding
// a long global history into a smaller number of bits; see P. Michaud's
// PPM-like predictor at CBP-1
class folded_history {
public:
  unsigned comp;
  int CLENGTH;
  int OLENGTH;
  int OUTPOINT;

  folded_history() {}

  void init(int original_length, int compressed_length) {
    comp = 0;
    OLENGTH = original_length;
    CLENGTH = compressed_length;
    OUTPOINT = OLENGTH % CLENGTH;
  }

  void update(uint8_t *h, int PT) {
    comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
    comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
    comp ^= (comp >> CLENGTH);
    comp = (comp) & ((1 << CLENGTH) - 1);
  }
};

class ientry // ITTAGE global table entry
{
public:
  uint64_t target;
  int8_t ctr;
  uint tag;
  int8_t u;

  ientry() {
    target = 0xdeadbeef;

    ctr = 0;
    u = 0;
    tag = 0;
  }
};

class IPREDICTOR {
public:
#define NHIST 8
#define MINHIST 2
#define MAXHIST 300
#define LOGG 10  /* logsize of the  banks in the  tagged ITTAGE tables */
#define TBITS 11 // tag width
#define NNN                                                                    \
  1 // number of extra entries allocated on an ITTAGE misprediction (1+NNN)

#define PHISTWIDTH 27 // width of the path history used in ITTAGE
#define UWIDTH 2      // u counter width on ITTAGE
#define CWIDTH 3      // predictor counter width on the ITTAGE tagged tables

  // the counter to chose between longest match and alternate prediction on
  // ITTAGE when weak confidence counters

  int8_t use_alt_on_na;
  long long GHIST;

  int TICK; // for the reset of the u counter
  uint8_t ghist[HISTBUFFERLENGTH];
  int ptghist;
  long long phist;                   // path history
  folded_history ch_i[NHIST + 1];    // utility for computing ITTAGE indices
  folded_history ch_t[2][NHIST + 1]; // utility for computing ITTAGE tags

  ientry *itable[NHIST + 1];
  int m[NHIST + 1];
  int TB[NHIST + 1];
  int logg[NHIST + 1];

  int GI[NHIST + 1]; // indexes to the different tables are computed only once
  uint GTAG[NHIST + 1]; // tags for the different tables are computed only once
  uint64_t pred_target; // prediction
  uint64_t alt_target;  // alternate  TAGEprediction
  uint64_t tage_target; // TAGE prediction

  uint64_t LongestMatchPred;
  int HitBank; // longest matching bank
  int AltBank; // alternate matching bank
  int Seed;    // for the pseudo-random number generator
  uint64_t target_inter;

  IPREDICTOR(void) { reinit(); }

  void reinit() {
    m[0] = 0;
    m[1] = MINHIST;
    m[NHIST] = MAXHIST;
    for (int i = 2; i <= NHIST; i++) {
      m[i] = (int)(((double)MINHIST * pow((double)(MAXHIST) / (double)MINHIST,
                                          (double)(i) / (double)NHIST)) +
                   0.5);
    }

    for (int i = 0; i <= NHIST; i++) {
      TB[i] = TBITS;
      logg[i] = LOGG;
    }

    for (int i = 0; i <= NHIST; i++)
      itable[i] = new ientry[(1 << LOGG)];

    for (int i = 0; i <= NHIST; i++) {
      ch_i[i].init(m[i], (logg[i]));
      ch_t[0][i].init(ch_i[i].OLENGTH, TB[i]);
      ch_t[1][i].init(ch_i[i].OLENGTH, TB[i] - 1);
    }

    Seed = 0;

    TICK = 0;
    phist = 0;
    Seed = 0;

    for (int i = 0; i < HISTBUFFERLENGTH; i++)
      ghist[0] = 0;
    ptghist = 0;
    use_alt_on_na = 0;
    GHIST = 0;
    ptghist = 0;
    phist = 0;
  }

  // F serves to mix path history: not very important impact

  int F(long long A, int size, int bank) {
    int A1, A2;
    A = A & ((1 << size) - 1);
    A1 = (A & ((1 << logg[bank]) - 1));
    A2 = (A >> logg[bank]);

    if (bank < logg[bank])
      A2 = ((A2 << bank) & ((1 << logg[bank]) - 1)) +
           (A2 >> (logg[bank] - bank));
    A = A1 ^ A2;
    if (bank < logg[bank])
      A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));
    return (A);
  }

  // gindex computes a full hash of PC, ghist and phist
  int gindex(unsigned int PC, int bank, long long hist, folded_history *ch_i) {
    int index;
    int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
    index = PC ^ (PC >> (abs(logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^
            F(hist, M, bank);

    return (index & ((1 << (logg[bank])) - 1));
  }

  //  tag computation
  uint16_t gtag(unsigned int PC, int bank, folded_history *ch0,
                folded_history *ch1) {
    int tag = (PC) ^ ch0[bank].comp ^ (ch1[bank].comp << 1);
    return (tag & ((1 << (TB[bank])) - 1));
  }

  // up-down saturating counter
  void ctrupdate(int8_t &ctr, bool taken, int nbits) {
    if (taken) {
      if (ctr < ((1 << (nbits - 1)) - 1))
        ctr++;
    } else {
      if (ctr > -(1 << (nbits - 1)))
        ctr--;
    }
  }

  // just a simple pseudo random number generator: use available information
  // to allocate entries  in the loop predictor
  int MYRANDOM() {
    Seed++;
    Seed ^= phist;
    Seed = (Seed >> 21) + (Seed << 11);
    Seed ^= ptghist;
    Seed = (Seed >> 10) + (Seed << 22);
    return (Seed);
  };

  //  ITTAGE PREDICTION: same code at fetch or retire time but the index and
  //  tags must recomputed
  uint64_t GetPrediction(uint64_t PC) {
    HitBank = -1;
    AltBank = -1;
    for (int i = 0; i <= NHIST; i++) {
      GI[i] = gindex(PC, i, phist, ch_i);
      GTAG[i] = gtag(PC, i, ch_t[0], ch_t[1]);
    }

    alt_target = 0;
    tage_target = 0;

    LongestMatchPred = 0;

    int AltConf = -4;
    int HitConf = -4;
    // Look for the bank with longest matching history
    for (int i = NHIST; i >= 0; i--) {
      if (itable[i][GI[i]].tag == GTAG[i]) {
        HitBank = i;
        HitConf = itable[HitBank][GI[HitBank]].ctr;
        LongestMatchPred = itable[HitBank][GI[HitBank]].target;
        break;
      }
    }

    // Look for the alternate bank
    for (int i = HitBank - 1; i >= 0; i--) {
      if (itable[i][GI[i]].tag == GTAG[i]) {
        alt_target = itable[i][GI[i]].target;
        AltBank = i;
        AltConf = itable[AltBank][GI[AltBank]].ctr;
        break;
      }
    }
    // computes the prediction and the alternate prediction

    if (HitBank > 0) {

      bool Huse_alt_on_na = (use_alt_on_na >= 0);
      if ((!Huse_alt_on_na) || (HitConf > 0) || (HitConf >= AltConf))
        tage_target = LongestMatchPred;
      else
        tage_target = alt_target;
    }
    if (AltBank < 0)
      tage_target = LongestMatchPred;

    return (tage_target);
  }

  void HistoryUpdate(uint64_t PC, uint64_t target, long long &X, int &Y,
                     folded_history *H, folded_history *G, folded_history *J) {

    int maxt = 3;
    int T = (PC >> 2) ^ (PC >> 6);
    int PATH = (target >> 2) ^ (target >> 6);

    for (int t = 0; t < maxt; t++) {
      bool DIR = (T & 1);
      T >>= 1;
      int PATHBIT = (PATH & 127);
      PATH >>= 1;
      // update  history
      Y--;
      ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
      X = (X << 1) ^ PATHBIT;

      for (int i = 1; i <= NHIST; i++) {

        H[i].update(ghist, Y);
        G[i].update(ghist, Y);
        J[i].update(ghist, Y);
      }
    }

    X = (X & ((1 << PHISTWIDTH) - 1));

    // END UPDATE  HISTORIES
  }

  void TrackOtherInst(uint64_t PC, uint64_t branchTarget) {

    HistoryUpdate(PC, branchTarget, phist, ptghist, ch_i, ch_t[0], ch_t[1]);
  }
  // PREDICTOR UPDATE

  void UpdatePredictor(uint64_t PC, uint64_t branchTarget) {

    // TAGE UPDATE
    bool ALLOC = ((tage_target != branchTarget) & (HitBank < NHIST));

    // do not allocate too often if the overall prediction is correct

    if (HitBank > 0)
      if (AltBank >= 0) {
        // Manage the selection between longest matching and alternate matching
        // for "pseudo"-newly allocated longest matching entry
        // this is extremely important for TAGE only, not that important when
        // the overall predictor is implemented
        bool PseudoNewAlloc = (itable[HitBank][GI[HitBank]].ctr <= 0);
        // an entry is considered as newly allocated if its prediction counter
        // is weak
        if (PseudoNewAlloc) {
          if (LongestMatchPred == branchTarget)
            ALLOC = false;
          // if it was delivering the correct prediction, no need to allocate a
          // new entry
          // even if the overall prediction was false
          if (LongestMatchPred != alt_target)
            if ((LongestMatchPred == branchTarget) ||
                (alt_target == branchTarget)) {
              ctrupdate(use_alt_on_na, (alt_target == branchTarget), ALTWIDTH);
            }
        }
      }

    if (ALLOC) {

      int T = NNN;
      int A = 1;
      if ((MYRANDOM() & 127) < 32)
        A = 2;
      int Penalty = 0;
      int NA = 0;
      int DEP = HitBank + A;
      for (int i = DEP; i <= NHIST; i++) {
        if (itable[i][GI[i]].u == 0) {
          itable[i][GI[i]].tag = GTAG[i];
          itable[i][GI[i]].target = branchTarget;
          itable[i][GI[i]].ctr = 0;
          NA++;
          if (T <= 0) {
            break;
          }
          i += 1;
          T -= 1;
        }

        else {
          Penalty++;
        }
      }

      TICK += (Penalty - 2 * NA);

      // just the best formula for the Championship:
      // In practice when one out of two entries are useful
      if (TICK < 0)
        TICK = 0;
      if (TICK >= BORNTICK) {

        for (int i = 0; i <= NHIST; i++)
          for (int j = 0; j < (1 << LOGG); j++)
            itable[i][j].u >>= 1;
        TICK = 0;
      }
    }

    // update predictions
    if (HitBank >= 0) {
      if (itable[HitBank][GI[HitBank]].ctr <= 0)
        if (LongestMatchPred != branchTarget)

        {
          if (alt_target == branchTarget)
            if (AltBank >= 0) {
              ctrupdate(itable[AltBank][GI[AltBank]].ctr,
                        (alt_target == branchTarget), CWIDTH);
            }
        }

      ctrupdate(itable[HitBank][GI[HitBank]].ctr,
                (LongestMatchPred == branchTarget), CWIDTH);
      if (LongestMatchPred != branchTarget)
        if (itable[HitBank][GI[HitBank]].ctr < 0)
          itable[HitBank][GI[HitBank]].target = branchTarget;
    }
    if (LongestMatchPred != alt_target)
      if (LongestMatchPred == branchTarget) {
        if (itable[HitBank][GI[HitBank]].u < (1 << UWIDTH) - 1)
          itable[HitBank][GI[HitBank]].u++;
      }
    // END TAGE UPDATE

    HistoryUpdate(PC, branchTarget, phist, ptghist, ch_i, ch_t[0], ch_t[1]);

    // END PREDICTOR UPDATE
  }
#undef NHIST
#undef MINHIST
#undef MAXHIST
#undef LOGG
#undef TBITS
#undef NNN
#undef PHISTWIDTH
#undef UWIDTH
#undef CWIDTH
};
#endif
