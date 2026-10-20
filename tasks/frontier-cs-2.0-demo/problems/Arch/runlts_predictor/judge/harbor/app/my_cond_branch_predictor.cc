#ifndef RUNLTS_H
#define RUNLTS_H

// cbp2025 submission #44 RUNLTS
// Author: Toru Koizumi (koizumi@nitech.ac.jp), Toshiki Maekawa, Masanari Mizuno, Maru Kuroki, Tomoaki Tsumura, Ryota Shioya

#include <array>
#include <assert.h>
#include <functional>
#include <inttypes.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <vector>

namespace nRUNLTS {

static constexpr int LogMaxNewlyCounters = 16;

static constexpr int NHIST = 23;
static constexpr double HistRate = 1.19;
static constexpr int Born2 = 18;
static constexpr int MINHIST = 6;

using UINT64 = uint64_t;
#define SC

#define ASSERT_EQ(a, b)                                                                                            \
    if (a != b) {                                                                                                  \
        std::cerr << "assertion [" #a "(" << a << ") == " #b "(" << b << ")] failed at " << __LINE__ << std::endl; \
    }
#define ASSERT_LT(a, b)                                                                                           \
    if (a >= b) {                                                                                                 \
        std::cerr << "assertion [" #a "(" << a << ") < " #b "(" << b << ")] failed at " << __LINE__ << std::endl; \
    }

struct BiasArgs {
    int HitBank;
    bool HighConf;
    bool LowConf;
    bool LongestMatchPred;
    bool alttaken;
    bool pred_inter;
};

struct SimpleComponent {
    // Constants
    size_t log_size;
    size_t ctr_width;
    std::function<size_t(UINT64 pc_hash, uint64_t full_hist, BiasArgs args)> index_func;
    int8_t ctr_min;
    int8_t ctr_max;

    // Actual Data
    std::vector<int8_t> values;

    SimpleComponent() = default; // default constructible
    SimpleComponent(
        size_t log_size, size_t ctr_width, decltype(index_func) index_func, int8_t (*init_val_gen)(size_t idx) = [](size_t i) -> int8_t { return i % 2 == 0 ? -1 : 0; })
        : log_size(log_size)
        , ctr_width(ctr_width)
        , index_func(std::move(index_func))
        , ctr_min(-(1 << (ctr_width - 1)))
        , ctr_max(~ctr_min)
        , values(1 << log_size)
    {
        for (size_t i = 0; i < values.size(); ++i)
            values.at(i) = init_val_gen(i);
    }

    int get_value(UINT64 pc_hash, uint64_t full_hist, BiasArgs args) const
    {
        ASSERT_LT(index_func(pc_hash, full_hist, args), values.size());
        int8_t ctr = values.at(index_func(pc_hash, full_hist, args));
        return 2 * ctr + 1; // convert to the balanced representation
    }

    void update(bool taken /* resolveDir */, UINT64 pc_hash, uint64_t full_hist, BiasArgs args)
    {
        ASSERT_LT(index_func(pc_hash, full_hist, args), values.size());
        int8_t& ctr = values.at(index_func(pc_hash, full_hist, args));
        if (taken)
            if (ctr < ctr_max)
                ++ctr;
        if (not taken)
            if (ctr > ctr_min)
                --ctr;
    }

    size_t storage_size() const { return ctr_width * values.size(); }
};

namespace sBiasNormal {
    // Parameter definition
    static constexpr size_t Width = 7; // PERCWIDTH
    static constexpr size_t LogSize = 10; // LOGBIAS

    static size_t index(UINT64 PC, uint64_t, BiasArgs args)
    {
        const bool LowConf = args.LowConf;
        const bool LongestMatchPred = args.LongestMatchPred;
        const bool alttaken = args.alttaken;
        const bool pred_inter = args.pred_inter;

        const bool uncertain = LowConf & (LongestMatchPred != alttaken);
        const size_t index_hash = PC ^ (PC >> 2);
        const size_t total_hash = (index_hash << 2) | (uncertain << 1) | pred_inter;
        return total_hash % (1 << LogSize);
    }
    static int8_t init_value(size_t i)
    {
        switch (i % 4) {
        case 0:
            return -32; // certain untaken
        case 1:
            return 31; // certain taken
        case 2:
            return -1; // uncertain untaken
        case 3:
            return 0; // uncertain taken
        }
        throw nullptr; // gcc is so fool
    }
};

namespace sBiasSkew {
    // Parameter definition
    static constexpr size_t Width = 7; // PERCWIDTH
    static constexpr size_t LogSize = 10; // LOGBIAS

    static size_t index(UINT64 PC, uint64_t, BiasArgs args)
    {
        const bool HighConf = args.HighConf;
        const bool pred_inter = args.pred_inter;

        const size_t index_hash = PC ^ (PC >> (LogSize - 2));
        const size_t total_hash = (index_hash << 2) | (HighConf << 1) | pred_inter;
        return total_hash % (1 << LogSize);
    }
    static int8_t init_value(size_t i)
    {
        switch (i % 4) {
        case 0:
            return -8; // not high confidence untaken
        case 1:
            return 7; // not high confidence taken
        case 2:
            return -32; // high confidence untaken
        case 3:
            return 31; // high confidence taken
        }
        throw nullptr; // gcc is so fool
    }
};

namespace sBiasBank {
    // Parameter definition
    static constexpr size_t Width = 7; // PERCWIDTH
    static constexpr size_t LogSize = 10; // LOGBIAS

    static size_t index(UINT64 PC, uint64_t, BiasArgs args)
    {
        const int HitBank = args.HitBank;
        const bool HighConf = args.HighConf;
        const bool alttaken = args.alttaken;
        const int pred_inter = args.pred_inter;

        const size_t pc_hash = PC ^ (PC >> 2);
        const size_t hit_bank_hash = HitBank / 3; // 0 <= HitBank <= 23, thus 0 <= HitBank/3 <= 7
        const size_t total_hash = (pc_hash << 6) | (hit_bank_hash << 3) | (alttaken << 2) | (HighConf << 1) | pred_inter;
        return total_hash % (1 << LogSize);
    }
    static int8_t init_value(size_t i)
    {
        switch (i % 4) {
        case 0:
            return -32; // high confidence untaken
        case 1:
            return 31; // high confidence taken
        case 2:
            return -1; // not high confidence untaken
        case 3:
            return 0; // not high confidence taken
        }
        throw nullptr; // gcc is so fool
    }
};

namespace sBrIMLI {
    // Parameter definition
    static constexpr size_t Width = 6; // PERCWIDTH
    static constexpr size_t LogSize = 10; // LOG_BrIMLI

    static size_t index(UINT64 PC, uint64_t BrIMLI, BiasArgs)
    {
        const size_t pc_hash = (PC >> 2) ^ (PC >> 8);
        const size_t total_hash = pc_hash ^ BrIMLI;
        return total_hash % (1 << LogSize);
    }
}

namespace sTaIMLI {
    // Parameter definition
    static constexpr size_t Width = 6; // PERCWIDTH
    static constexpr size_t LogSize = 11; // LOG_TaIMLI

    static size_t index(UINT64 PC, uint64_t TaIMLI, BiasArgs)
    {
        const size_t pc_hash = (PC << 1) ^ (PC >> 6);
        const size_t total_hash = pc_hash ^ TaIMLI;
        return total_hash % (1 << LogSize);
    }
}

struct WeightGroup {
    // Constants
    static constexpr size_t Width = 6; // EWIDTH
    static constexpr size_t LogSize = 3; // LOGSIZEUPS = LOGSIZEUP/2
    static constexpr int8_t ctr_min = -(1 << (Width - 1));
    static constexpr int8_t ctr_max = ~ctr_min;

    // Actual Data
    std::array<int8_t, 1 << LogSize> weight_ctr;
    std::vector<SimpleComponent> components; // We can add components after call the constructor

    WeightGroup(int init_val)
    {
        std::fill(weight_ctr.begin(), weight_ctr.end(), init_val);
    }

    size_t index_for_weight(UINT64 PC) const
    {
        return (PC ^ (PC >> 2)) % weight_ctr.size();
    }

    int get_value(UINT64 pc_hash, uint64_t full_hist, BiasArgs args) const
    {
        int sum = 0;
        for (const auto& c : components)
            sum += c.get_value(pc_hash, full_hist, args);
        return sum;
    }

    int get_weighted_value(UINT64 pc_hash, uint64_t full_hist, BiasArgs args) const
    {
        int weight = weight_ctr[index_for_weight(pc_hash)] >= 0 ? 2 : 1; // This seems wrong because GGEHL uses both PC and PC<<1|pred_inter as weight index arguments
        return weight * get_value(pc_hash, full_hist, args);
    }

    void update(int LSUM, bool resolveDir, UINT64 pc_hash, uint64_t full_hist, BiasArgs args)
    {
        // Weight update
        const int sum = get_value(pc_hash, full_hist, args);
        const int weighted_sum = get_weighted_value(pc_hash, full_hist, args);

        const int LSUM_if_weight_is_1 = LSUM - weighted_sum + 1 * sum;
        const int LSUM_if_weight_is_2 = LSUM - weighted_sum + 2 * sum;
        const bool pred_if_weight_is_1 = LSUM_if_weight_is_1 >= 0;
        const bool pred_if_weight_is_2 = LSUM_if_weight_is_2 >= 0;

        if (pred_if_weight_is_1 != pred_if_weight_is_2) {
            const bool our_pred = sum >= 0;
            const bool useful = our_pred == resolveDir;
            int8_t& w = weight_ctr[index_for_weight(pc_hash)];
            if (useful)
                if (w < ctr_max)
                    ++w;
            if (not useful)
                if (w > ctr_min)
                    --w;
        }

        // updates each component
        for (auto& c : components)
            c.update(resolveDir, pc_hash, full_hist, args);
    }

    int get_extra_weight(UINT64 PC) const { return weight_ctr[index_for_weight(PC)] >= 0; }

    size_t storage_size() const
    {
        size_t total = 0;
        total += Width * weight_ctr.size();
        for (auto& c : components)
            total += c.storage_size();
        return total;
    }
};

struct wBias : WeightGroup {
    wBias()
        : WeightGroup(4)
    {
        WeightGroup::components.emplace_back(sBiasNormal::LogSize, sBiasNormal::Width, sBiasNormal::index, sBiasNormal::init_value);
        WeightGroup::components.emplace_back(sBiasSkew::LogSize, sBiasSkew::Width, sBiasSkew::index, sBiasSkew::init_value);
        WeightGroup::components.emplace_back(sBiasBank::LogSize, sBiasBank::Width, sBiasBank::index, sBiasBank::init_value);
    }
} bias_components;

struct wGEHL : WeightGroup {
    size_t MaxLength;
    wGEHL(std::vector<size_t> Lengths, size_t LogSize, size_t Width)
        : MaxLength(Lengths.at(0))
        , WeightGroup(7)
    {
        for (size_t i = 0; i < Lengths.size(); ++i) {
            const size_t log_size = LogSize - (i >= Lengths.size() - 2); // last two components are half size
            const auto index_func = [i, log_size, length = Lengths.at(i)](UINT64 pc_hash, uint64_t full_hist, BiasArgs) { return hash_func_template(pc_hash, full_hist, length, i) % (1 << log_size); };
            WeightGroup::components.emplace_back(log_size, Width, index_func);
        }
    }

    static size_t hash_func_template(UINT64 pc_hash, uint64_t BHIST, int length, int i)
    {
        const uint64_t bhist = BHIST & ((1ull << length) - 1);
        size_t hash = pc_hash;
        hash ^= bhist;
        hash ^= bhist >> (8 - i);
        hash ^= bhist >> (16 - 2 * i);
        hash ^= bhist >> (24 - 3 * i);
        hash ^= bhist >> (32 - 3 * i); // Trick in the TAGE-SC-L (2016)
        hash ^= bhist >> (40 - 4 * i);
        return hash;
    }
};

wGEHL global_GEHL_components {
    { 40, 24, 10 }, // Gm
    11, // LOGGNB
    6, // PERCWIDTH
};

wGEHL path_GEHL_components {
    { 16, 9 }, // Pm
    10, // LOGPNB
    6, // PRECWIDTH
};

wGEHL local1_GEHL_components {
    { 18, 11, 6, 3 }, // Lm
    11, // LOGLNB
    6, // PERCWIDTH
};

wGEHL local2_GEHL_components {
    { 21, 16, 11, 6 }, // Sm
    11, // LOGSNB
    6, // PERCWIDTH
};

wGEHL local3_GEHL_components {
    { 19, 14, 9, 4 }, // Tm
    11, // LOGTNB
    6, // PERCWIDTH
};

wGEHL call_stack_GEHL_components {
    { 47, 31, 18, 10, 5 }, // Cm
    11, // LOGCNB
    6, // PERCWIDTH
};

struct IMLI_WeightGroup {
    // Constants
    static constexpr size_t Width = 6; // EWIDTH
    static constexpr size_t LogSize = 8; // LOGSIZEUPS = LOGSIZEUP/2 = 3 is not sufficient
    static constexpr int8_t ctr_min = -(1 << (Width - 1));
    static constexpr int8_t ctr_max = ~ctr_min;

    // Actual Data
    std::array<int8_t, 1 << LogSize> weight_ctr;
    std::vector<SimpleComponent> components;

    IMLI_WeightGroup()
    {
        std::fill(weight_ctr.begin(), weight_ctr.end(), 0);
        components.emplace_back(sBrIMLI::LogSize, sBrIMLI::Width, sBrIMLI::index);
        components.emplace_back(sTaIMLI::LogSize, sTaIMLI::Width, sTaIMLI::index);
    }

    size_t index_for_weight(UINT64 PC) const
    {
        return (PC ^ (PC >> 2)) % weight_ctr.size();
    }

    int get_value(UINT64 PC, uint64_t BrIMLI, uint64_t TaIMLI) const
    {
        int sum = 0;
        sum += components.at(0).get_value(PC, BrIMLI, {});
        sum += components.at(1).get_value(PC, TaIMLI, {});
        return sum;
    }

    int get_weighted_value(uint64_t PC, uint64_t BrIMLI, uint64_t TaIMLI) const
    {
        int weight = 1 + get_extra_weight(PC);
        return weight * get_value(PC, BrIMLI, TaIMLI);
    }

    void update(int LSUM, bool resolveDir, uint64_t PC, uint64_t BrIMLI, uint64_t TaIMLI)
    {
        // Weight update
        const int sum = get_value(PC, BrIMLI, TaIMLI);
        const int weighted_sum = get_weighted_value(PC, BrIMLI, TaIMLI);

        const int LSUM_if_weight_is_1 = LSUM - weighted_sum + 1 * sum;
        const int LSUM_if_weight_is_3 = LSUM - weighted_sum + 3 * sum; // IMLIs sometimes provide highly correlated information, thus weight is not 2x but 3x.
        const bool pred_if_weight_is_1 = LSUM_if_weight_is_1 >= 0;
        const bool pred_if_weight_is_3 = LSUM_if_weight_is_3 >= 0;

        if (pred_if_weight_is_1 != pred_if_weight_is_3) {
            const bool our_pred = sum >= 0;
            const bool useful = our_pred == resolveDir;
            int8_t& w = weight_ctr[index_for_weight(PC)];
            if (useful)
                if (w < ctr_max)
                    ++w;
            if (not useful)
                if (w > ctr_min)
                    --w;
        }

        // updates each component
        for (int i = 0; i < 1 + get_extra_weight(PC); ++i) {
            components.at(0).update(resolveDir, PC, BrIMLI, {});
            components.at(1).update(resolveDir, PC, TaIMLI, {});
        }
    }

    int get_extra_weight(UINT64 PC) const { return 2 * (weight_ctr[index_for_weight(PC)] >= 0); }

    size_t storage_size() const
    {
        size_t total = 0;
        total += Width * weight_ctr.size();
        for (auto& c : components)
            total += c.storage_size();
        return total;
    }
} IMLI_components;

namespace sLocal1 {
    static constexpr size_t FeatureSize = 256;
    size_t get_index(uint64_t PC) { return (PC ^ (PC >> 2)) % FeatureSize; }
}

namespace sLocal2 {
    static constexpr size_t FeatureSize = 16;
    size_t get_index(uint64_t PC) { return (PC ^ (PC >> 5)) % FeatureSize; }
}

namespace sLocal3 {
    static constexpr size_t FeatureSize = 16;
    size_t get_index(uint64_t PC) { return (PC ^ (PC >> 11)) % FeatureSize; } // 11 is LOGTNB
}

namespace sCallStack {
    static constexpr size_t FeatureSize = 8;
}

// playing with putting more weights (x2)  on some of the SC components
// playing on using different update thresholds on SC
// update threshold for the statistical corrector
#define LOGSIZEUP 6 // not worth increasing
#define WIDTHRES 12
#define WIDTHRESP 8
int updatethreshold;
int Pupdatethreshold[(1 << LOGSIZEUP)]; // size is fixed by LOGSIZEUP
#define INDUPD (PC ^ (PC >> 2)) & ((1 << LOGSIZEUP) - 1)

// The two counters used to choose between TAGE and SC on Low Conf SC
int8_t FirstH, SecondH;

#define CONFWIDTH 7 // for the counters in the choser
#define HISTBUFFERLENGTH 8192 // we use a 8K entries history buffer to store the branch history (this allows us to explore using history length up to 8K)

// utility class for index computation
// this is the cyclic shift register for folding
// a long global history into a smaller number of bits; see P. Michaud's PPM-like predictor at CBP-1

class gentry // TAGE global table entry
{
public:
    int8_t ctr;
    uint tag;
    // useful_or_newly_alloc means...
    // ctr == 3,5,7 : useful (u)
    // ctr == 1 : newly allocated (novel state)
    bool useful_or_newly_alloc;

    bool is_useful() const { return abs(2 * ctr + 1) != 1 & useful_or_newly_alloc; }
    bool is_newly_alloc() const { return abs(2 * ctr + 1) == 1 & useful_or_newly_alloc; }

    gentry()
    {
        ctr = 0;
        useful_or_newly_alloc = false;
        tag = 0;
    }
};

#define NBANKLOW 9 // number of banks in the shared bank-interleaved for the low history lengths
#define NBANKHIGH 25 // number of banks in the shared bank-interleaved for the  history lengths

int SizeTable[NHIST + 1];

#define BORN 5 // below BORN in the table for low history lengths, >= BORN in the table for high history lengths,

#define LOGG 11 /* logsize of the  banks in the  tagged TAGE tables */
#define TBITS 9 // minimum width of the tags  (low history lengths), +4 for high history lengths

#define NNN 2 // number of extra entries allocated on a TAGE misprediction (1+NNN)
#define HYSTSHIFT 2 // bimodal hysteresis shared by 4 entries
#define LOGB 17 // log of number of entries in bimodal predictor

std::array<int8_t, 1ull << LOGB> bim_pred;
std::array<int8_t, 1ull << LOGB - HYSTSHIFT> bim_hyst;

#define PHISTWIDTH 27 // width of the path history used in TAGE
#define UWIDTH 1 // u counter width on TAGE (2 bits not worth the effort for a 512 Kbits predictor 0.2 %)
#define CWIDTH 3 // predictor counter width on the TAGE tagged tables

// the counter(s) to chose between longest match and alternate prediction on TAGE when weak counters
#define LOGSIZEUSEALT 4
#define ALTWIDTH 5
#define SIZEUSEALT (1 << (LOGSIZEUSEALT))
#define INDUSEALT (((((pv.HitBank - 1) / 8) << 1) + pv.AltConf) % (SIZEUSEALT - 1))
int8_t use_alt_on_na[SIZEUSEALT];

// For the TAGE predictor
gentry* gtable[NHIST + 1]; // tagged TAGE tables
int m[NHIST + 1];
int TB[NHIST + 1];
int logg[NHIST + 1];

// Monitoring blocking frequency by u (useful)
static int BORNTICK = 1024;
int TICK; // [0, 1024)

void TICK_update(int NumberOfBlocks, int NumberOfAllocations)
{
    TICK += (NumberOfBlocks - 2 * NumberOfAllocations);
    if (TICK < 0) {
        TICK = 0;
    }
    if (TICK >= BORNTICK) {
        // Low Bank
        for (int j = 0; j < SizeTable[1]; ++j) {
            if (gtable[1][j].is_useful())
                gtable[1][j].useful_or_newly_alloc = false;
        }
        // High Bank
        for (int j = 0; j < SizeTable[BORN]; ++j) {
            if (gtable[BORN][j].is_useful())
                gtable[BORN][j].useful_or_newly_alloc = false;
        }
        // Reset TICK
        TICK = 0;
    }
}

uint64_t Seed; // for the pseudo-random number generator

class folded_history {
public:
    unsigned comp;
    int CLENGTH;
    int OLENGTH;
    int OUTPOINT;

    folded_history()
    {
    }

    void init(int original_length, int compressed_length)
    {
        comp = 0;
        OLENGTH = original_length;
        CLENGTH = compressed_length;
        OUTPOINT = OLENGTH % CLENGTH;
    }

    void update(std::array<uint8_t, HISTBUFFERLENGTH>& h, int PT)
    {
        comp = (comp << 1) ^ h[PT & (HISTBUFFERLENGTH - 1)];
        comp ^= h[(PT + OLENGTH) & (HISTBUFFERLENGTH - 1)] << OUTPOINT;
        comp ^= (comp >> CLENGTH);
        comp = (comp) & ((1 << CLENGTH) - 1);
    }
};
using tage_index_t = std::array<folded_history, NHIST + 1>;
using tage_tag_t = std::array<folded_history, NHIST + 1>;

struct cbp_hist_t {
    // Checkpoint information
    tage_index_t ch_i;
    std::array<tage_tag_t, 2> ch_t;
    int perceptron_sum_at_prediction;

    // Begin Conventional Histories
    uint64_t phist; // path history
    std::array<uint8_t, HISTBUFFERLENGTH> ghist;
    int ptghist;

    // For SC
    uint64_t GHIST; // backward history
    uint64_t fphist; // forward taken path history
    std::array<uint64_t, sLocal1::FeatureSize> L_shist;
    std::array<uint64_t, sLocal2::FeatureSize> S_slhist;
    std::array<uint64_t, sLocal3::FeatureSize> T_slhist;
    std::array<uint64_t, sCallStack::FeatureSize> C_hist;
    size_t CallStackPtr = 0;

    uint64_t last_backward_target = 0;
    uint64_t last_backward_pc = 0;
    uint64_t BrIMLI = 0;
    uint64_t TaIMLI = 0;

    uint64_t& local1_hist(uint64_t PC) { return L_shist.at(sLocal1::get_index(PC)); }
    uint64_t& local2_hist(uint64_t PC) { return S_slhist.at(sLocal2::get_index(PC)); }
    uint64_t& local3_hist(uint64_t PC) { return T_slhist.at(sLocal3::get_index(PC)); }
    uint64_t& call_stack_hist() { return C_hist.at(CallStackPtr); }
    uint64_t local1_hist(uint64_t PC) const { return L_shist.at(sLocal1::get_index(PC)); }
    uint64_t local2_hist(uint64_t PC) const { return S_slhist.at(sLocal2::get_index(PC)); }
    uint64_t local3_hist(uint64_t PC) const { return T_slhist.at(sLocal3::get_index(PC)); }
    uint64_t call_stack_hist() const { return C_hist.at(CallStackPtr); }

    cbp_hist_t()
    {
    }
};

void print_predictorsize()
{
    int STORAGESIZE = 0;

    int bim = (1 << LOGB) + (1 << (LOGB - HYSTSHIFT));
    printf("(BIM %d) ", bim);
    STORAGESIZE += bim;

    int tagged = 0;
    tagged += NBANKHIGH * (1 << (logg[BORN])) * (CWIDTH + UWIDTH + TB[BORN]);
    tagged += NBANKLOW * (1 << (logg[1])) * (CWIDTH + UWIDTH + TB[1]);

    tagged += (SIZEUSEALT)*ALTWIDTH;
    tagged += m[NHIST];
    tagged += PHISTWIDTH;
    tagged += 10; // the TICK counter

    tagged += LogMaxNewlyCounters * 2; // Allocation throttling counters (NewlyDecay and NewlyUseful)

    printf("(Tagged %d) ", tagged);
    STORAGESIZE += tagged;

    int inter = 0;
#ifdef SC
    inter += WIDTHRES;
    inter = WIDTHRESP * ((1 << LOGSIZEUP)); // the update threshold counters

    inter += bias_components.storage_size();
    inter += global_GEHL_components.storage_size();
    inter += global_GEHL_components.MaxLength; // global backward history for SC
    inter += path_GEHL_components.storage_size();
    inter += path_GEHL_components.MaxLength; // forward taken pathhistory

    inter += local1_GEHL_components.storage_size();
    inter += sLocal1::FeatureSize * local1_GEHL_components.MaxLength;
    inter += local2_GEHL_components.storage_size();
    inter += sLocal2::FeatureSize * local2_GEHL_components.MaxLength;
    inter += local3_GEHL_components.storage_size();
    inter += sLocal3::FeatureSize * local3_GEHL_components.MaxLength;

    inter += call_stack_GEHL_components.storage_size();
    inter += sCallStack::FeatureSize * call_stack_GEHL_components.MaxLength;
    inter += 3; // CallStackPtr

    inter += IMLI_components.storage_size();
    inter += 64; // last_backward_target
    inter += 64; // last_backward_pc
    inter += sBrIMLI::LogSize; // BrIMLI
    inter += sTaIMLI::LogSize; // TaIMLI

    printf("(TraditionalSC %d) ", inter);

    inter += 2 * CONFWIDTH; // the 2 counters in the choser

    printf("(TotalSC %d) ", inter);
    STORAGESIZE += inter;
#endif

    printf("\nStorageSizeKiB = %f (%d bits)", STORAGESIZE / 8192., STORAGESIZE);
}

struct PredRelatedVariables {
    int LSUM;
    bool MedConf;
    bool AltConf; // Confidence on the alternate prediction
    int8_t BIM;
    bool tage_pred; // TAGE prediction
    bool alttaken; // alternate  TAGEprediction
    bool LongestMatchPred;
    int HitBank; // longest matching bank
    int AltBank; // alternate matching bank
    bool pred_inter;

    bool LowConf;
    bool HighConf;

    // state set by predict
    int GI[NHIST + 1]; // indexes to the different tables are computed only once
    uint GTAG[NHIST + 1]; // tags for the different tables are computed only once
    int BI; // index of the bimodal table

    //
    int THRES;
    //

    bool final_prediction;
    bool SCPRED;

    int chooser() const
    {
        if (pred_inter != SCPRED) {
            if (HighConf) {
                if (abs(LSUM) < THRES / 4) {
                    return 0; // use pred_inter
                } else if (abs(LSUM) < THRES / 2) {
                    return 2; // use SecondH<0 ? SCPRED : pred_inter
                } else {
                    return 3; // use SCPRED
                }
            } else if (MedConf) {
                if (abs(LSUM) < THRES / 4) {
                    return 1; // use FirstH<0 ? SCPRED : pred_inter
                } else {
                    return 3; // use SCPRED
                }
            } else {
                return 3; // use SCPRED
            }
        } else {
            return 3; // use SCPRED (== pred_inter)
        }
    }
};

class RUNLTS {
    int NewlyDecay, NewlyUseful = 4;

public:
    cbp_hist_t active_hist; // running history always updated accurately
    // checkpointed history. Can be accesed using the inst-id(seq_no/piece)
    std::unordered_map<uint64_t /*key*/, cbp_hist_t /*val*/> pred_time_histories;
    PredRelatedVariables pv;

    RUNLTS(void)
    {
        init_histories(active_hist);
        print_predictorsize();
    }

    void setup()
    {
    }

    void terminate()
    {
    }

    uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const
    {
        assert(piece < 16);
        return (seq_no << 4) | (piece & 0x000F);
    }

    int CalcNextHistLen(int before, int current, double rate)
    {
        int a = current + (current - before + 2); // interval is an arithmetic sequence
        int b = int(current * rate / 2 + 0.5) * 2; // geometric sequence
        return std::max(a, b);
    }

    void init_histories(cbp_hist_t& current_hist)
    {
        m[1] = MINHIST;
        for (int i = 2; i <= NHIST; ++i) {
            double rate = std::max(HistRate, HistRate + 0.1 * (i - Born2));
            m[i] = CalcNextHistLen(m[i - 2], m[i - 1], rate);
        }
        for (int i = 1; i <= NHIST; i++) {
            TB[i] = TBITS + 4 * (i >= BORN);
            logg[i] = LOGG;
        }

        gtable[1] = new gentry[NBANKLOW * (1 << LOGG)];
        SizeTable[1] = NBANKLOW * (1 << LOGG);

        gtable[BORN] = new gentry[NBANKHIGH * (1 << LOGG)];
        SizeTable[BORN] = NBANKHIGH * (1 << LOGG);

        for (int i = BORN + 1; i <= NHIST; i++)
            gtable[i] = gtable[BORN];
        for (int i = 2; i <= BORN - 1; i++)
            gtable[i] = gtable[1];

        for (int i = 1; i <= NHIST; i++) {
            current_hist.ch_i[i].init(m[i], (logg[i]));
            current_hist.ch_t[0][i].init(current_hist.ch_i[i].OLENGTH, TB[i]);
            current_hist.ch_t[1][i].init(current_hist.ch_i[i].OLENGTH, TB[i] - 1);
        }

        TICK = 0;
        current_hist.phist = 0;
        Seed = 0;

        for (int i = 0; i < HISTBUFFERLENGTH; i++)
            current_hist.ghist[i] = 0;
        updatethreshold = 35;

        for (int i = 0; i < (1 << LOGSIZEUP); i++)
            Pupdatethreshold[i] = 0;

        std::fill(bim_pred.begin(), bim_pred.end(), 0);
        std::fill(bim_hyst.begin(), bim_hyst.end(), 1);

        for (int i = 0; i < SIZEUSEALT; i++) {
            use_alt_on_na[i] = 0;
        }
        for (int i = 0; i < sLocal1::FeatureSize; i++) {
            current_hist.L_shist[i] = 0;
        }
        for (int i = 0; i < sLocal2::FeatureSize; i++) {
            current_hist.S_slhist[i] = 0;
        }
        for (int i = 0; i < sLocal3::FeatureSize; i++) {
            current_hist.T_slhist[i] = 0;
        }

        current_hist.GHIST = 0;
        current_hist.ptghist = 0;
        current_hist.phist = 0;
    } // end init_histories

    // the index functions for the tagged tables uses path history as in the OGEHL predictor
    // F serves to mix path history: not very important impact
    int F(uint64_t A, int size, int bank) const
    {
        int A1, A2;
        A = A & ((1 << size) - 1);
        A1 = (A & ((1 << logg[bank]) - 1));
        A2 = (A >> logg[bank]);

        if (bank < logg[bank])
            A2 = ((A2 << bank) & ((1 << logg[bank]) - 1)) + (A2 >> (logg[bank] - bank));

        A = A1 ^ A2;

        if (bank < logg[bank])
            A = ((A << bank) & ((1 << logg[bank]) - 1)) + (A >> (logg[bank] - bank));

        return A;
    }

    // gindex computes a full hash of PC, ghist and phist
    // int gindex (unsigned int PC, int bank, uint64_t hist, const folded_history * ch_i) const
    int gindex(unsigned int PC, int bank, uint64_t hist, const tage_index_t& ch_i) const
    {
        int index;
        int M = (m[bank] > PHISTWIDTH) ? PHISTWIDTH : m[bank];
        index = PC ^ (PC >> (abs(logg[bank] - bank) + 1)) ^ ch_i[bank].comp ^ F(hist, M, bank);

        return (index & ((1 << (logg[bank])) - 1));
    }

    //  tag computation
    uint16_t gtag(unsigned int PC, int bank, const tage_tag_t& tag_0_array, const tage_tag_t& tag_1_array) const
    {
        int tag = (PC) ^ tag_0_array[bank].comp ^ (tag_1_array[bank].comp << 1);
        return (tag & ((1 << (TB[bank])) - 1));
    }

    // up-down saturating counter
    void ctrupdate(int& ctr, bool taken, int nbits)
    {
        if (taken) {
            if (ctr < ((1 << (nbits - 1)) - 1))
                ctr++;
        } else {
            if (ctr > -(1 << (nbits - 1)))
                ctr--;
        }
    }
    void ctrupdate(int8_t& ctr, bool taken, int nbits)
    {
        if (taken) {
            if (ctr < ((1 << (nbits - 1)) - 1))
                ctr++;
        } else {
            if (ctr > -(1 << (nbits - 1)))
                ctr--;
        }
    }

    // just a simple pseudo random number generator: use available information
    //  to allocate entries  in the loop predictor
    int MYRANDOM()
    {
        Seed++;
        Seed ^= active_hist.phist;
        Seed = (Seed >> 21) + (Seed << 11);
        Seed ^= (int64_t)active_hist.ptghist;
        Seed = (Seed >> 10) + (Seed << 22);
        return (Seed & 0xFFFFFFFF);
    }

    bool gentry_tag_match(int i) { return gtable[i][pv.GI[i]].tag == pv.GTAG[i]; }
    int8_t& gentry_ctr(int i) { return gtable[i][pv.GI[i]].ctr; }
    bool gentry_pred(int i) { return gentry_ctr(i) >= 0; }
    int gentry_magnitude(int i) { return abs(2 * gentry_ctr(i) + 1); }
    bool gentry_already_trained(int i) { return gentry_magnitude(i) > 1; } // 3or5or7
    bool gentry_newly_allocated(int i) { return gentry_magnitude(i) == 1; }

    //  TAGE PREDICTION: same code at fetch or retire time but the index and tags must recomputed
    void Prepare_GI_GTAG_BI(UINT64 PC, const cbp_hist_t& hist_to_use)
    {
        // Calculate tag and index
        for (int i = 1; i <= NHIST; ++i) {
            size_t gi = gindex(PC, i, hist_to_use.phist, hist_to_use.ch_i); // phist and ghist folded into Index-width bits
            uint gt = gtag(PC, i, hist_to_use.ch_t[0], hist_to_use.ch_t[1]); // ghist folded into Tag-width bits and Tag-width -1 bits
            pv.GI[i] = gi;
            pv.GTAG[i] = gt;
        }

        // HighBank
        {
            // We can use only m[BORN] bits for bank shuffling entropy
            int T = (PC >> 2 ^ (hist_to_use.phist & ((1ull << m[BORN]) - 1))) % NBANKHIGH;
            for (int i = BORN; i <= NHIST; i++) {
                pv.GI[i] += (T << LOGG);
                T = (T + 1) % NBANKHIGH; // Adjoining bank
            }
        }

        // Low Bank
        {
            int T = (PC >> 2 ^ (hist_to_use.phist & ((1 << m[1]) - 1))) % NBANKLOW;
            for (int i = 1; i <= BORN - 1; i++) {
                pv.GI[i] += (T << LOGG);
                T = (T + 1) % NBANKLOW;
            }
        }

        // Calculate Bim index
        pv.BI = (PC ^ (PC >> 2)) & ((1 << LOGB) - 1);
    }

    void Tagepred(UINT64 PC, const cbp_hist_t& hist_to_use)
    {
        // Calculate tag and index
        Prepare_GI_GTAG_BI(PC, hist_to_use);

        // Search Bim
        const bool bim = bim_pred[pv.BI] > 0;
        pv.BIM = (bim_pred[pv.BI] << 1) | bim_hyst[pv.BI >> HYSTSHIFT]; // Not a shared-split counter form... just a hysteresis-shared 2bc

        // Look for the bank with longest matching history
        pv.HitBank = 0;
        pv.tage_pred = bim;
        pv.LongestMatchPred = bim;
        pv.HighConf = (pv.BIM == 0) || (pv.BIM == 3);
        pv.LowConf = (pv.BIM == 1) || (pv.BIM == 2);
        pv.MedConf = false;
        for (int i = NHIST; i > 0; i--) {
            if (gentry_tag_match(i)) {
                pv.HitBank = i;
                pv.LongestMatchPred = gentry_pred(pv.HitBank);
                pv.HighConf = gentry_magnitude(pv.HitBank) >= (1 << CWIDTH) - 1; // 7
                pv.MedConf = gentry_magnitude(pv.HitBank) == 5;
                pv.LowConf = gentry_magnitude(pv.HitBank) == 1;
                break;
            }
        }

        // Look for the alternate bank
        pv.AltBank = 0;
        pv.alttaken = bim;
        pv.AltConf = (pv.BIM == 0) || (pv.BIM == 3);
        for (int i = pv.HitBank - 1; i > 0; i--) {
            if (gentry_tag_match(i)) {
                pv.AltBank = i;
                pv.alttaken = gentry_pred(pv.AltBank);
                pv.AltConf = gentry_already_trained(pv.AltBank); // 3 or 5 or 7
                break;
            }
        }

        // computes the prediction and the alternate prediction
        if (pv.HitBank > 0) {
            // if the entry is recognized as a newly allocated entry and
            // USE_ALT_ON_NA is positive  use the alternate prediction
            bool Huse_alt_on_na = (use_alt_on_na[INDUSEALT] >= 0);
            if (not gentry_newly_allocated(pv.HitBank)) {
                pv.tage_pred = pv.LongestMatchPred; // Not newly allocated: choose the longest match
            } else if (Huse_alt_on_na) {
                pv.tage_pred = pv.alttaken; // Newly allocated and use_alt_on_na says it's not high conf: choose the second longest match
            } else {
                pv.tage_pred = pv.LongestMatchPred; // Otherwise, choose the longest match
            }
        }
    }

    bool predict_using_given_hist(uint64_t seq_no, uint8_t piece, UINT64 PC, cbp_hist_t& hist_to_use, const bool pred_time_predict)
    {
        // computes the TAGE table addresses and the partial tags
        Tagepred(PC, hist_to_use);
#ifndef SC
        return pv.tage_pred;
#endif
        pv.pred_inter = pv.tage_pred; // We don't use the Loop Predictor, thus, pred_inter is the same as tage_pred.

        int extra_weight = 0;
        pv.LSUM = 0;

        // Compute the SC prediction
        pv.LSUM += bias_components.get_weighted_value(PC, 0ull, { pv.HitBank, pv.HighConf, pv.LowConf, pv.LongestMatchPred, pv.alttaken, pv.pred_inter });
        pv.LSUM += global_GEHL_components.get_weighted_value((PC << 1) + pv.pred_inter, hist_to_use.GHIST, {});
        pv.LSUM += path_GEHL_components.get_weighted_value(PC, hist_to_use.fphist, {});
        pv.LSUM += local1_GEHL_components.get_weighted_value(PC, hist_to_use.local1_hist(PC), {});
        pv.LSUM += local2_GEHL_components.get_weighted_value(PC, hist_to_use.local2_hist(PC), {});
        pv.LSUM += local3_GEHL_components.get_weighted_value(PC, hist_to_use.local3_hist(PC), {});
        pv.LSUM += call_stack_GEHL_components.get_weighted_value(PC, hist_to_use.call_stack_hist(), {});
        pv.LSUM += IMLI_components.get_weighted_value(PC, hist_to_use.BrIMLI, hist_to_use.TaIMLI);
        pv.SCPRED = (pv.LSUM >= 0);

        // just  an heuristic if the respective contribution of component groups can be multiplied by 2 or not
        const int base_threshold = (updatethreshold >> 1) + Pupdatethreshold[INDUPD];

        extra_weight += 2 * bias_components.get_extra_weight(PC);
        extra_weight += 2 * global_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * path_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * local1_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * local2_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * local3_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * call_stack_GEHL_components.get_extra_weight(PC);
        extra_weight += 2 * IMLI_components.get_extra_weight(PC);
        pv.THRES = base_threshold + 6 * extra_weight;

        // Minimal benefit in trying to avoid accuracy loss on low confidence SC prediction and  high/medium confidence on TAGE
        //  but just uses 2 counters 0.3 % MPKI reduction
        const int chooser = pv.chooser();
        switch (chooser) {
        case 0:
            return pv.pred_inter;
        case 1:
            return FirstH < 0 ? pv.SCPRED : pv.pred_inter;
        case 2:
            return SecondH < 0 ? pv.SCPRED : pv.pred_inter;
        case 3:
            return pv.SCPRED;
        default:
            throw nullptr;
        }
    }

    void HistoryUpdate(UINT64 PC, int brtype, bool pred_taken, bool taken, UINT64 nextPC)
    {
        auto& X = active_hist.phist;
        auto& Y = active_hist.ptghist;

        auto& H = active_hist.ch_i;
        auto& G = active_hist.ch_t[0];
        auto& J = active_hist.ch_t[1];

        // special treatment for indirect  branchs;
        int maxt = 2;
        if (brtype & 1) // conditional
            maxt = 2;
        else if ((brtype & 2))
            maxt = 3;

        if (brtype & 1) {
            active_hist.GHIST = (active_hist.GHIST << 1) + (taken & (nextPC < PC)); // backward taken only dir history (low entropy)
            active_hist.local1_hist(PC) = (active_hist.local1_hist(PC) << 1) | taken;
            active_hist.local2_hist(PC) = ((active_hist.local2_hist(PC) << 1) | taken) ^ (PC & 15);
            active_hist.local3_hist(PC) = (active_hist.local3_hist(PC) << 1) | taken;
        }

        int T = ((PC ^ (PC >> 2))) ^ taken;
        int PATH = PC ^ (PC >> 2) ^ (PC >> 4);

        for (int t = 0; t < maxt; t++) {
            bool DIR = (T & 1);
            T >>= 1;
            int PATHBIT = (PATH & 127);
            PATH >>= 1;
            // update  history
            Y--; // ptghist
            active_hist.ghist[Y & (HISTBUFFERLENGTH - 1)] = DIR;
            X = (X << 1) ^ PATHBIT; // phist

            // updates to folded histories
            for (int i = 1; i <= NHIST; i++) {
                H[i].update(active_hist.ghist, Y);
                G[i].update(active_hist.ghist, Y);
                J[i].update(active_hist.ghist, Y);
            }
        }
        X = (X & ((1 << PHISTWIDTH) - 1));

        // forward taken path history
        if (nextPC > PC && taken)
            active_hist.fphist = (active_hist.fphist << 3) ^ (nextPC >> 2) ^ (PC >> 1);

        // for call stack history
        active_hist.call_stack_hist() <<= 1;
        active_hist.call_stack_hist() |= taken;
        if (brtype & 4) { // call
            active_hist.CallStackPtr += 1;
            active_hist.CallStackPtr %= sCallStack::FeatureSize;
            active_hist.call_stack_hist() = 0;
        }
        if (brtype & 8) { // return
            active_hist.CallStackPtr += sCallStack::FeatureSize - 1;
            active_hist.CallStackPtr %= sCallStack::FeatureSize;
        }

        // IMLI
        if (taken && nextPC < PC && (brtype & 2) == 0) {
            int prime[18] = { 1, 1, 3, 7, 13, 31, 61, 127, 251, 509, 1021, 2039, 4093, 8191, 16381, 32749, 65521, 131071 }; // not exceeding the power of two
            if (active_hist.last_backward_target / 128 == nextPC / 128) {
                active_hist.TaIMLI = (active_hist.TaIMLI + 1) % prime[sTaIMLI::LogSize];
            } else {
                active_hist.TaIMLI = 0;
            }

            if (active_hist.last_backward_pc / 128 == PC / 128) {
                active_hist.BrIMLI = (active_hist.BrIMLI + 1) % (1ull << sBrIMLI::LogSize);
            } else {
                active_hist.BrIMLI = 0;
            }
            active_hist.last_backward_target = nextPC;
            active_hist.last_backward_pc = PC;
        }

    } // END UPDATE  HISTORIES

    // PREDICTOR UPDATE
    void baseupdate(bool Taken)
    {
        int inter = pv.BIM;
        if (Taken) {
            if (inter < 3)
                inter += 1;
        } else if (inter > 0)
            inter--;
        bim_pred[pv.BI] = inter >> 1;
        bim_hyst[pv.BI >> HYSTSHIFT] = (inter & 1);
    }

    void TAGE_do_allocation(bool resolveDir)
    {
        int NumberOfBlocks = 0;
        int NumberOfAllocations = 0;

        const bool AggressiveAllocation = NewlyDecay <= NewlyUseful * 2;
        const bool ModestAllocation = NewlyDecay > NewlyUseful * 4;
        int RestAllocations = AggressiveAllocation ? NNN + 2 : ModestAllocation ? NNN
                                                                                : NNN + 1;

        const int skip = (MYRANDOM() & 127) < 32 ? 1 : 0;
        const int provider_bank_next = pv.HitBank + 1;
        const int start_bank = provider_bank_next + skip;

        int last_alloc_index = -1;

        for (int i = start_bank; i <= NHIST; ++i) {
            if (AggressiveAllocation && gtable[i][pv.GI[i]].is_newly_alloc()) {
                gtable[i][pv.GI[i]].useful_or_newly_alloc = false;
                ++NewlyDecay;
                if (NewlyDecay >= 1 << LogMaxNewlyCounters) {
                    NewlyDecay >>= 1;
                    NewlyUseful >>= 1;
                }
                --RestAllocations;
            } else if (gtable[i][pv.GI[i]].is_useful() == 0) {
                if (abs(2 * gtable[i][pv.GI[i]].ctr + 1) <= 3) {
                    gtable[i][pv.GI[i]].tag = pv.GTAG[i];
                    gtable[i][pv.GI[i]].ctr = (resolveDir) ? 0 : -1;
                    if (gtable[i][pv.GI[i]].is_newly_alloc()) {
                        gtable[i][pv.GI[i]].useful_or_newly_alloc = false;
                        ++NewlyDecay;
                        if (NewlyDecay >= 1 << LogMaxNewlyCounters) {
                            NewlyDecay >>= 1;
                            NewlyUseful >>= 1;
                        }
                    }
                    last_alloc_index = i;
                    ++NumberOfAllocations;
                    RestAllocations -= 1;
                    if (RestAllocations <= 0) {
                        break;
                    }
                    i += i < 3 ? 1 : i < Born2 ? 2
                                               : 0;
                } else {
                    if (gtable[i][pv.GI[i]].ctr > 0)
                        gtable[i][pv.GI[i]].ctr--;
                    else
                        gtable[i][pv.GI[i]].ctr++;
                }
            } else {
                ++NumberOfBlocks;
            }
        }

        if (last_alloc_index != -1)
            gtable[last_alloc_index][pv.GI[last_alloc_index]].useful_or_newly_alloc = true;

        TICK_update(NumberOfBlocks, NumberOfAllocations);
    }

    void SC_update(UINT64 PC, bool resolveDir, const cbp_hist_t& hist_to_use)
    {
        pv.SCPRED = pv.LSUM >= 0;

        // FirstH and SecondH is meta predictors (tagged vs SC)
        // Train FirstH and SecondH
        const int chooser = pv.chooser();
        switch (chooser) {
        case 0:
            break; // do nothing
        case 1:
            ctrupdate(FirstH, (pv.pred_inter == resolveDir), CONFWIDTH);
            break;
        case 2:
            ctrupdate(SecondH, (pv.pred_inter == resolveDir), CONFWIDTH);
            break;
        case 3:
            break; // do nothing
        default:
            throw nullptr;
        }

        bool sum_at_prediction_is_weak = abs(hist_to_use.perceptron_sum_at_prediction) < pv.THRES;
        bool sum_at_train_is_weak = abs(pv.LSUM) < pv.THRES;
        bool mispred_at_prediction = (hist_to_use.perceptron_sum_at_prediction >= 0) != resolveDir;
        bool mispred_at_train = (pv.LSUM >= 0) != resolveDir;

        auto threshold_update = [PC](int up) {
            Pupdatethreshold[INDUPD] = std::min((1 << (WIDTHRESP - 1)) - 1, std::max(-(1 << (WIDTHRESP - 1)), Pupdatethreshold[INDUPD] + up));
            updatethreshold = std::min((1 << (WIDTHRES - 1)) - 1, std::max(-(1 << (WIDTHRES - 1)), updatethreshold + up));
        };

        if (mispred_at_prediction and not mispred_at_train) {
            threshold_update(+6); // FTL++-like threshold update. There are strong perturbations.
        }
        if (mispred_at_prediction and mispred_at_train) {
            threshold_update(+1); // Mispredictions; increase the threshold (there are many perturbations)
        }
        if (not mispred_at_prediction and sum_at_prediction_is_weak and sum_at_train_is_weak) {
            threshold_update(-1); // A weak prediction is correct; decrease the threshold (there are small perturbations)
        }
        // Do not decrease the threshold in the cases wehere sum_at_prediction_is_weak but not sum_at_train_is_weak; it may be a loop begining.

        if (mispred_at_train or sum_at_train_is_weak) {
            bias_components.update(pv.LSUM, resolveDir, PC, 0ull, { pv.HitBank, pv.HighConf, pv.LowConf, pv.LongestMatchPred, pv.alttaken, pv.pred_inter });
            global_GEHL_components.update(pv.LSUM, resolveDir, (PC << 1) + pv.pred_inter, hist_to_use.GHIST, {});
            path_GEHL_components.update(pv.LSUM, resolveDir, PC, hist_to_use.fphist, {});
            local1_GEHL_components.update(pv.LSUM, resolveDir, PC, hist_to_use.local1_hist(PC), {});
            local2_GEHL_components.update(pv.LSUM, resolveDir, PC, hist_to_use.local2_hist(PC), {});
            local3_GEHL_components.update(pv.LSUM, resolveDir, PC, hist_to_use.local3_hist(PC), {});
            call_stack_GEHL_components.update(pv.LSUM, resolveDir, PC, hist_to_use.call_stack_hist(), {});
            IMLI_components.update(pv.LSUM, resolveDir, PC, hist_to_use.BrIMLI, hist_to_use.TaIMLI);

        }
    }

    void TAGE_allocation(UINT64 PC, bool resolveDir, bool pred_taken)
    {
        bool ALLOC = ((pv.tage_pred != resolveDir) & (pv.HitBank < NHIST));

        if (pv.HitBank > 0) {
            const bool PseudoNewAlloc = gentry_newly_allocated(pv.HitBank);
            if (PseudoNewAlloc) {
                // The longest match entry will provide correct prediction if it is trained now.
                // (even if use_alt_on_na or SC cause a misprediction). Thus, do not allocate.
                if (pv.LongestMatchPred == resolveDir)
                    ALLOC = false;

                // Train use_alt_on_na
                if (pv.LongestMatchPred != pv.alttaken) {
                    ctrupdate(use_alt_on_na[INDUSEALT], (pv.alttaken == resolveDir), ALTWIDTH);
                }
            }
        }

        // Do not allocate too often if the overall prediction is correct
        if (pred_taken == resolveDir)
            if ((MYRANDOM() & 31) != 0)
                ALLOC = false;

        if (ALLOC) {
            TAGE_do_allocation(resolveDir);
        }
    }

    void TAGE_train(UINT64 PC, bool resolveDir)
    {
        if (pv.HitBank > 0) {
            // Only when the longest-match entry mispredicts, train the second longest entry. It is not 'useful'.
            if (gentry_magnitude(pv.HitBank) == 1 && pv.LongestMatchPred != resolveDir) {
                if (pv.AltBank > 0) {
                    if (gtable[pv.AltBank][pv.GI[pv.AltBank]].is_newly_alloc())
                        gtable[pv.AltBank][pv.GI[pv.AltBank]].useful_or_newly_alloc = false;
                    ctrupdate(gentry_ctr(pv.AltBank), resolveDir, CWIDTH);
                    if (abs(2 * gtable[pv.AltBank][pv.GI[pv.AltBank]].ctr + 1) == 1)
                        gtable[pv.AltBank][pv.GI[pv.AltBank]].useful_or_newly_alloc = false; // Ensure u == false when abs(2*ctr+1) == 1 for our novel states
                } else {
                    baseupdate(resolveDir);
                }
            }

            if (gtable[pv.HitBank][pv.GI[pv.HitBank]].is_newly_alloc()) {
                if (pv.LongestMatchPred == resolveDir) {
                    ++NewlyUseful;
                }
                gtable[pv.HitBank][pv.GI[pv.HitBank]].useful_or_newly_alloc = false;
                if (NewlyUseful >= 1 << LogMaxNewlyCounters) {
                    NewlyDecay >>= 1;
                    NewlyUseful >>= 1;
                }
            }

            // Train the longest-match entry
            // The counter come 0 or -1 states, remove 'useful'.
            ctrupdate(gentry_ctr(pv.HitBank), resolveDir, CWIDTH);
            if (gentry_magnitude(pv.HitBank) == 1)
                gtable[pv.HitBank][pv.GI[pv.HitBank]].useful_or_newly_alloc = false;

            // The second-longest-match entry has high confidence
            // and provides the same direction and it's correct, remove 'useful'.
            if (pv.alttaken == resolveDir && pv.AltBank > 0 && gentry_magnitude(pv.AltBank) == 7)
                if (pv.LongestMatchPred == resolveDir && gtable[pv.HitBank][pv.GI[pv.HitBank]].is_useful())
                    gtable[pv.HitBank][pv.GI[pv.HitBank]].useful_or_newly_alloc = false;
        } else {
            // The longest match is Bim, train Bim.
            baseupdate(resolveDir);
        }

        // The longest-match entry revokes the second-longest-match entry and it's correct, add 'useful'.
        if (pv.LongestMatchPred != pv.alttaken && pv.LongestMatchPred == resolveDir)
            gtable[pv.HitBank][pv.GI[pv.HitBank]].useful_or_newly_alloc = true;
    }

    void update(UINT64 PC, bool resolveDir, bool pred_taken, UINT64 nextPC, const cbp_hist_t& hist_to_use)
    {
#ifdef SC
        SC_update(PC, resolveDir, hist_to_use);
#endif
        TAGE_allocation(PC, resolveDir, pred_taken);
        TAGE_train(PC, resolveDir);
    } // END PREDICTOR UPDATE

    // Trampoline functions
    bool predict(uint64_t seq_no, uint8_t piece, UINT64 PC)
    {
        pred_time_histories.emplace(get_unique_inst_id(seq_no, piece), active_hist); // checkpoint current hist
        const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, active_hist, true /*pred_time_predict*/);
        pred_time_histories.at(get_unique_inst_id(seq_no, piece)).perceptron_sum_at_prediction = pv.LSUM;
        return pred_taken;
    }
    void history_update(uint64_t seq_no, uint8_t piece, UINT64 PC, int brtype, bool pred_taken, bool taken, UINT64 nextPC)
    {
        HistoryUpdate(PC, brtype, pred_taken, taken, nextPC);
    }
    void TrackOtherInst(UINT64 PC, int brtype, bool pred_taken, bool taken, UINT64 nextPC)
    {
        HistoryUpdate(PC, brtype, pred_taken, taken, nextPC);
    }
    void update(uint64_t seq_no, uint8_t piece, UINT64 PC, bool resolveDir, bool predDir, UINT64 nextPC)
    {
        const auto pred_hist_key = get_unique_inst_id(seq_no, piece);
        auto& pred_time_history = pred_time_histories.at(pred_hist_key);
        const bool pred_taken = predict_using_given_hist(seq_no, piece, PC, pred_time_history, false /*pred_time_predict*/);
        update(PC, resolveDir, pred_taken, nextPC, pred_time_history);
        pred_time_histories.erase(pred_hist_key);
    }
    void decode_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg)
    {
    }
    void execute_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg, uint64_t value)
    {
    }
};
// =================
// Predictor End
// =================

#undef UINT64

} // namespace nRUNLTS

#include "my_cond_branch_predictor.h"

void CBP2025_RUNLTS::setup()
{
    p_impl = std::make_shared<nRUNLTS::RUNLTS>();
    p_impl->setup();
}
bool CBP2025_RUNLTS::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) { return p_impl->predict(seq_no, piece, PC); }
void CBP2025_RUNLTS::update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir, bool predDir, uint64_t nextPC) { p_impl->update(seq_no, piece, PC, resolveDir, predDir, nextPC); }
void CBP2025_RUNLTS::history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, int brtype, bool pred_dir, bool resolve_dir, uint64_t nextPC) { p_impl->history_update(seq_no, piece, PC, brtype, pred_dir, resolve_dir, nextPC); }
void CBP2025_RUNLTS::TrackOtherInst(uint64_t PC, int brtype, bool pred_dir, bool resolve_dir, uint64_t nextPC) { p_impl->TrackOtherInst(PC, brtype, resolve_dir, pred_dir, nextPC); }
void CBP2025_RUNLTS::terminate() { p_impl->terminate(); }
void CBP2025_RUNLTS::decode_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg) { p_impl->decode_notify(seq_no, piece, dst_reg); }
void CBP2025_RUNLTS::execute_notify(uint64_t seq_no, uint8_t piece, uint64_t dst_reg, uint64_t value) { p_impl->execute_notify(seq_no, piece, dst_reg, value); }

#endif
