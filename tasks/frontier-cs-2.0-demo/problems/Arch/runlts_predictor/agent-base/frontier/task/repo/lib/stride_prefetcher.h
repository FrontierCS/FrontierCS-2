#pragma once

#include <cassert>
#include <vector>
#include <deque>
#include <map>
#include <algorithm>
//#include <optional>

#define DEF_ENUM(ENUM, NAME) _DEF_ENUM(ENUM, NAME)
#define _DEF_ENUM(ENUM, NAME)                          \
   case ENUM::NAME:                                    \
      stream << #NAME ;                                 \
      break;
//Ref: Effective Hardware-Based Data Prefetching for High-Performance Processors
// https://ieeexplore.ieee.org/document/381947/

struct PrefetchTrainingInfo
{
    uint64_t pc;
    uint64_t address;
    uint64_t size;
    bool miss;

    friend std::ostream &operator<<(std::ostream &stream, const PrefetchTrainingInfo &info)
    {
        stream << "PC: "<< std::hex  << info.pc << " Address: "<< std::hex  << info.address << " Size: "<< std::hex  << info.size << " Miss? " << info.miss;
        return stream;
    }
};

enum class PrefetcherState
{
    Invalid,
    Initial,
    Transient,
    SteadyState,
    NoPrediction,
    NumPrefetcherStates
};

static std::ostream& operator<<(std::ostream& stream, const PrefetcherState& s)
{
    switch(s){
     DEF_ENUM(PrefetcherState, Invalid)
     DEF_ENUM(PrefetcherState, Initial)
     DEF_ENUM(PrefetcherState, Transient)
     DEF_ENUM(PrefetcherState, SteadyState)
     DEF_ENUM(PrefetcherState, NoPrediction)
    default:
      assert(false);
    };
    return stream;
}

constexpr uint64_t NUM_RPT_ENTRIES = 1024;
constexpr uint64_t PREFETCH_MULTIPLIER = 2; // 2 because when we lookahead, we are 1 behind, so need next(next(access))
constexpr int PF_QUEUE_SIZE = 32;
constexpr uint64_t CACHE_LINE_MASK = ~63lu;
constexpr uint64_t PF_MUST_ISSUE_BEFORE_CYCLES = 8;


struct RPTEntry
{
    PrefetcherState state =  PrefetcherState::Invalid;
    uint64_t tag = 0xdeadbeef;
    uint64_t prev_address = 0xdeadbeef;
    uint64_t current_address = 0xdeadbeef;
    int64_t stride = -1;
    uint64_t lru= 0;
    uint64_t index = -1;

    RPTEntry() =default;
    RPTEntry(PrefetcherState st_, uint64_t t_ , uint64_t p_ , uint64_t c_ , int64_t s_, uint64_t l_, uint64_t i_)
    :state(st_)
    ,tag(t_)
    ,prev_address(p_)
    ,current_address(c_)
    ,stride(s_)
    ,lru(l_)
    ,index(i_)
    {}

    friend std::ostream& operator<<(std::ostream& stream, const RPTEntry& e)
    {
        stream << "Index:" <<std::hex << e.index << " State " << e.state << " Tag: " << std::hex << e.tag << " Prev: " << std::hex << e.prev_address << " Cur: " << std::hex << e.current_address << " Stride: " << std::hex << e.stride << " LRU: " << e.lru;
        return stream;

    }
};

enum class CacheLevel
{
    Invalid,
    L1,
    L2,
    L3
};

struct Prefetch
{

    explicit Prefetch(uint64_t a_, uint64_t cycle)
    : address(a_)
    , cycle_generated(cycle)
    {}
    Prefetch() = default;

    friend std::ostream &operator<<(std::ostream &stream, const Prefetch& pf)
    {
        stream << "[PF: Address: "<< std::hex  << pf.address << std::dec << ", cyclegen: " << pf.cycle_generated << "]";
        return stream;
    }

    uint64_t address = 0xdeadbeef;
    uint64_t cycle_generated = ~0lu;
    //CacheLevel level;
};

class StridePrefetcher
{
   public:
    void init(const uint64_t n)
    {
        for(uint64_t i = 0; i < n; i++)
        {
            //Initialize LRU
            rpt[i].index = i;
            rpt[i].lru = i;
        }
        //Clear queue of generated prefetches
        queue.clear();
    }

    StridePrefetcher()
    {
        init(NUM_RPT_ENTRIES);
    }

    uint64_t victim_way()
    {
        auto entry = std::find_if(rpt.begin(), rpt.end(), [](RPTEntry& e){ return !e.lru; });
        assert((entry != rpt.end()) && "Must find a valid victim way ");
        spdlog::debug("Prefetch: Found victim entry : {}", *entry);

        return entry->index;
    }

    void update_lru(uint64_t index)
    {
        spdlog::debug("Updating LRU Index: {}", index);
        auto& lru_way = rpt[index];
        std::for_each(rpt.begin(), rpt.end(), [&lru_way](RPTEntry &e) { if(e.lru > lru_way.lru){ --e.lru;} });
        lru_way.lru = (NUM_RPT_ENTRIES - 1);
    }

    // Prefetches will be generated when the load is fetched as in "Effective Hardware-Based Data Prefetching for High-Performance Processors"
    // However because we train immediately, there is no need for a count variable.
    void lookahead(uint64_t la_pc, uint64_t cycle)
    {
        auto entry = std::find_if(rpt.begin(), rpt.end(), [la_pc](RPTEntry& e){ return e.tag == la_pc; });
        if(entry == rpt.end())
        {
            return;
        }
        else if(entry->state == PrefetcherState::SteadyState)
        {
            generate(*entry, cycle);
        }
    }

    void train(const PrefetchTrainingInfo & info)
    {
        spdlog::debug("Prefetcher: Training on LD {}", info);
        auto entry = std::find_if(rpt.begin(), rpt.end(), [&info](RPTEntry& e){ return e.tag == info.pc; });
        if(entry == rpt.end())
        {
            //Establish a new entry
            auto victim_index = victim_way();
            auto& victim_entry = rpt[victim_index];
            victim_entry.state = PrefetcherState::Initial;
            victim_entry.tag = info.pc;
            victim_entry.prev_address = 0xdeadbeef;
            victim_entry.current_address = info.address;
            victim_entry.stride = 0;
            spdlog::debug("Prefetcher: Overwriting entry now in Initial STate : {}", victim_entry);
            update_lru(victim_index);
        }
        else
        {
            switch(entry->state){
                case PrefetcherState::Initial:
                {
                    int64_t stride = info.address - entry->current_address;
                    if (stride == entry->stride)
                    {
                        entry->state = PrefetcherState::SteadyState;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: Initial->SteadyState: {}", *entry);
                    }else{
                        entry->stride = stride;
                        entry->state = PrefetcherState::Transient;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: Initial->Transient: {}", *entry);
                    }
                }
                break;
                case PrefetcherState::Transient:
                {
                    int64_t stride = info.address - entry->current_address;
                    if (stride == entry->stride)
                    {
                        entry->state = PrefetcherState::SteadyState;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: Transient->SteadyState: {}", *entry);
                    }
                    else
                    {
                        entry->state = PrefetcherState::NoPrediction;
                        entry->stride = stride;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: Transient->NoPrediction: {}", *entry);
                    }
                }
                break;
                case PrefetcherState::SteadyState:
                {
                    int64_t stride = info.address - entry->current_address;
                    if (stride == entry->stride)
                    {
                        entry->state = PrefetcherState::SteadyState;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                    }else{
                        entry->state = PrefetcherState::Initial;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: SteadyState->Initial: {}", *entry);
                    }
                }
                break;
                case PrefetcherState::NoPrediction:
                {
                    int64_t stride = info.address - entry->current_address;
                    if (stride == entry->stride)
                    {
                        entry->state = PrefetcherState::Transient;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: NoPrediction->Transient: {}", *entry);
                    }
                    else
                    {
                        entry->state = PrefetcherState::NoPrediction;
                        entry->stride = stride;
                        entry->prev_address = entry->current_address;
                        entry->current_address = info.address;
                        spdlog::debug("Prefetcher: NoPrediction->NoPrediction: {}", *entry);
                    }
                }
                break;
                case PrefetcherState::Invalid:
                {
                    assert(false && "Unpexted state");
                }
                break;
                default:
                  assert(false);
                  break;
            };
            if(entry->stride != 0)
            {
                // Let entries with stride 0 age out
                update_lru(entry->index);
            }
        }

        //Update stats
        ++stat_trainings;
    }

    void generate(RPTEntry& entry, uint64_t cycle)
    {
        if(entry.stride == 0)
        {
            ++stat_stride_zero;
            return;
        }

        Prefetch pf{entry.current_address + entry.stride * PREFETCH_MULTIPLIER, cycle};
        spdlog::debug("Prefetcher: Queuing a new prefetch: {} Entry {}", pf, entry);

        auto it = std::find_if(queue.begin(), queue.end(), [&](const Prefetch & qpf)
        {
            return (qpf.address & CACHE_LINE_MASK) == (pf.address & CACHE_LINE_MASK);
        });

        if(it == queue.end())
        {
            queue.push_back(pf);
            
            // Make extra sure PF are sorted by generation order from oldest to youngest
            std::sort(queue.begin(), queue.end(), [](const Prefetch & lhs, const Prefetch & rhs)
            {
                if(lhs.cycle_generated != rhs.cycle_generated)
                {
                    return lhs.cycle_generated < rhs.cycle_generated;
                }
                if(lhs.address != rhs.address)
                {
                    return lhs.address < rhs.address;
                }
                return false;
            });

            ++stat_generated;
        }
        else
        {
            spdlog::debug("Prefetcher: Dropping pf: {} because already in pf queue", pf);
            ++stat_duplicate_pf_filtered;
        }
        
    }

    bool issue(Prefetch& p, uint64_t cycle)
    {
        while(!queue.empty() && (queue.front().cycle_generated + PF_MUST_ISSUE_BEFORE_CYCLES) < cycle)
        {
            spdlog::debug("Dropping pf because too old (created at cycle {}, current fetch cycle {})", queue.front().cycle_generated, cycle);
            ++stat_dropped_untimely_pf;
            queue.pop_front();
        }

        if(!queue.empty())
        {
            p = queue.front();
            if(p.cycle_generated <= cycle)
            {
                queue.pop_front();
                ++stat_issued;
                return true;
            }
            spdlog::debug("Giving up for now because not created yet (created at cycle {}, current fetch cycle {})", p.cycle_generated, cycle);
            return false;
        }
        return false;
    }

    void put_back(const Prefetch & p)
    {
        ++stat_put_back;
        queue.push_front(p);
    }

    uint64_t get_oldest_pf_cycle() const
    {
        if(queue.empty())
        {
            return MAX_CYCLE;
        }
        else 
        {
            return queue.front().cycle_generated;
        }
    }

    void print_stats()
    {
        std::cout << "Num Trainings :" << std::dec << stat_trainings  <<std::endl;
        std::cout << "Num Prefetches generated :" << stat_generated << std::endl;
        std::cout << "Num Prefetches issued :" << stat_issued << std::endl;
        std::cout << "Num Prefetches filtered by PF queue :" << stat_duplicate_pf_filtered << std::endl;
        std::cout << "Num untimely prefetches dropped from PF queue :" << stat_dropped_untimely_pf << std::endl;
        std::cout << "Num prefetches not issued LDST contention :" << stat_put_back << std::endl;
        std::cout << "Num prefetches not issued stride 0 :" << stat_stride_zero << std::endl;
    }
    private:
    std::array<RPTEntry, NUM_RPT_ENTRIES> rpt;
    uint64_t lru_info;

    //Queue to store generated prefetches
    std::deque<Prefetch> queue;
    //Stats
    uint64_t stat_trainings = 0;
    uint64_t stat_generated = 0;
    uint64_t stat_issued = 0;
    uint64_t stat_duplicate_pf_filtered = 0;
    uint64_t stat_dropped_untimely_pf = 0;
    uint64_t stat_put_back = 0;
    uint64_t stat_stride_zero = 0;

};
