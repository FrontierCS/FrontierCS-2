#include <spectrum/common/thread-util.hpp>
#include <thread>
#include <pthread.h>
#include <string>
#include <sstream>
#include <fmt/core.h>
#include <iostream>

void PinRoundRobin(std::thread& thread, unsigned rotate_id) {
    #include <pthread.h>
    auto core_id    = rotate_id % std::thread::hardware_concurrency();
    cpu_set_t   cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET (core_id, &cpu_set);
    auto rc = pthread_setaffinity_np(
        thread.native_handle(),
        sizeof(cpu_set_t), &cpu_set
    );
    if (rc != 0) {
        std::stringstream ss;
        ss << "cannot pin thread-" << thread.get_id()
           << " to core "          << core_id;       
        throw std::runtime_error(ss.str());
    }
}

void PinRoundRobin(std::jthread& thread, unsigned rotate_id) {
    #include <pthread.h>
    auto core_id    = rotate_id % std::thread::hardware_concurrency();
    cpu_set_t   cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET (core_id, &cpu_set);
    auto rc = pthread_setaffinity_np(
        thread.native_handle(),
        sizeof(cpu_set_t), &cpu_set
    );
    if (rc != 0) {
        std::stringstream ss;
        ss << "cannot pin thread-" << thread.get_id()
           << " to core "          << core_id;       
        throw std::runtime_error(ss.str());
    }
}
