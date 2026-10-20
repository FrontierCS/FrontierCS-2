#include <spectrum/protocol/abstraction.hpp>
#include <spectrum/protocol/spectrum.hpp>
#include <spectrum/protocol/sparkle.hpp>
#include <spectrum/protocol/aria-fb.hpp>
#include <spectrum/protocol/serial.hpp>
#include <spectrum/protocol/calvin.hpp>
#include <spectrum/protocol/dummy.hpp>
#include <spectrum/protocol/sparkle-partial.hpp>
#include <spectrum/protocol/sparkle-pre-sched.hpp>
#include <spectrum/protocol/spectrum-no-partial.hpp>
#include <spectrum/protocol/spectrum-no-partial-pre-sched.hpp>
#include <spectrum/protocol/spectrum-pre-sched.hpp>
#include <spectrum/protocol/spectrum-sched.hpp>
#include <spectrum/protocol/spectrum-cache.hpp>
#include <spectrum/workload/abstraction.hpp>
#include <spectrum/workload/smallbank.hpp>
#include <spectrum/workload/ycsb.hpp>
#include <spectrum/workload/tpcc.hpp>
#include "macros.hpp"
#include <ranges>
#include <iostream>

// expanding macros to assignments and use them later
#define ASSGIN_ARGS_HELPER(X, ...) __VA_OPT__(auto NAME(__VA_ARGS__) = (X);)
#define FILLIN_ARGS_HELPER(X, ...) __VA_OPT__(, NAME(__VA_ARGS__))
#define ASSGIN_ARGS(...)           FOR_EACH(ASSGIN_ARGS_HELPER, __VA_ARGS__, _)
#define FILLIN_ARGS(X, ...)        NAME(__VA_ARGS__, _) FOR_EACH(FILLIN_ARGS_HELPER, __VA_ARGS__, _)

// throw error
#define THROW(...)   throw std::runtime_error(std::string{fmt::format(__VA_ARGS__)})

// declare some helper macros for argument parser (by default, we name the token iterator by 'iter')
#define INT     to<size_t>  (*++iter)
#define DOUBLE  to<double>  (*++iter)
#define BOOL    to<bool>    (*++iter)
#define EVMTYPE ParseEVMType(*++iter)

using namespace spectrum;
using namespace std::chrono_literals;
using namespace std::chrono;

static auto split(std::basic_string_view<char> s) {
    auto iter = s | std::ranges::views::split(':')
    | std::ranges::views::transform([](auto&& str) { return std::string_view(&*str.begin(), std::ranges::distance(str)); });
    auto toks = std::vector<std::string>();
    for (auto x: iter) {
        toks.push_back(std::string{x});
    }
    return toks;
}

template<typename T>
static T to(std::basic_string_view<char> s) {
    std::stringstream sstream(std::string{s});
    T result; sstream >> result;
    return result;
}

template<>
bool to<bool>(std::basic_string_view<char> s) {
    if (s == "TRUE")    { return true; }
    if (s == "FALSE")   { return false; }
    THROW("cannot recognize ({}) as boolean should be either TRUE or FALSE", s);
}

template<>
milliseconds to<milliseconds>(std::basic_string_view<char> s) {
    std::stringstream is(std::string{s});
	static const std::unordered_map<std::string, milliseconds> suffix {
        {"ms", 1ms}, {"s", 1s}, {"m", 1min}, {"h", 1h}};
    unsigned n {};
    std::string unit;
    if (is >> n >> unit) {
        try {
            return duration_cast<milliseconds>(n * suffix.at(unit));
        } catch (const std::out_of_range&) {
            std::cerr << "ERROR: Invalid unit specified\n";
        }
    } else {
        std::cerr << "ERROR: Could not convert to numeric value\n";
    }
	return milliseconds{};
}

inline std::unique_ptr<Workload> ParseWorkload(const char* arg) {
    auto args = split(arg);
    auto name = *args.begin();
    auto dist = (size_t) (std::distance(args.begin(), args.end()) - 1);
    auto iter = args.begin();
    // map each option to an argparser
    #define OPT(X, Y...) if (name == #X) { \
        auto n = (size_t) COUNT(Y);        \
        ASSGIN_ARGS(Y);                    \
        if (dist != n) THROW("workload {} has {} args -- ({}), but we found {} args", #X, n, #Y, dist); \
        return static_cast<std::unique_ptr<Workload>>(std::make_unique<X>(FILLIN_ARGS(Y))); \
    };
    OPT(Smallbank, INT, DOUBLE)
    OPT(YCSB     , INT, DOUBLE)
    OPT(TPCC     , INT, INT)
    #undef OPT
    // fallback to an error
    THROW("unknown workload option ({})", std::string{name});
}

inline std::unique_ptr<Protocol> ParseProtocol(const char* arg, Workload& workload, Statistics& statistics) {
    auto args = split(arg);
    auto name = *args.begin();
    auto dist = (size_t) (std::distance(args.begin(), args.end()) - 1);
    auto iter = args.begin();
    // map each option to an argparser
    #define OPT(X, Y...) if (name == #X) { \
        auto n = (size_t) COUNT(Y);        \
        ASSGIN_ARGS(Y);                    \
        if (dist != n) THROW("protocol {} has {} args -- ({}), but we found {} args", #X, n, #Y, dist); \
        return static_cast<std::unique_ptr<Protocol>>(std::make_unique<X>(workload, statistics, FILLIN_ARGS(Y)));  \
    };
    OPT(Aria,               INT, INT, INT, BOOL)
    OPT(Sparkle,            INT, INT)
    OPT(SparklePartial,     INT, INT, EVMTYPE)
    OPT(SparklePreSched,    INT, INT, EVMTYPE)
    OPT(Spectrum,           INT, INT, EVMTYPE)
    OPT(SpectrumSched,      INT, INT, EVMTYPE)
    OPT(SpectrumCache,      INT, INT, EVMTYPE)
    OPT(SpectrumPreSched,   INT, INT, EVMTYPE)
    OPT(SpectrumNoPartial,  INT, INT, EVMTYPE)
    OPT(SpectrumNoPartialPreSched, INT, INT, EVMTYPE)
    OPT(Serial,             EVMTYPE,  INT)
    OPT(Calvin,             INT, INT, INT)
    OPT(Dummy,              INT, INT, EVMTYPE)
    // Calvin num_threads, num_dispatchers(default 1), table_partitions
    #undef OPT
    // fallback to an error
    THROW("unknown protocol option ({})", std::string{name});
}

// remove helper macros
#undef INT
#undef BOOL
#undef DOUBLE
#undef EVMTYPE
#undef FILLIN_ARGS
#undef FILLIN_ARGS_HELPER
#undef ASSGIN_ARGS
#undef ASSGIN_ARGS_HELPER
#undef THROW
