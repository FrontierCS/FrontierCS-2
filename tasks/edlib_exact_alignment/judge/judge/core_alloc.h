#pragma once
extern "C" void edlib_core_enter(void* frame);
extern "C" void edlib_core_leave();
struct EdlibCoreAllocationScope {
    explicit EdlibCoreAllocationScope(void* frame) { edlib_core_enter(frame); }
    ~EdlibCoreAllocationScope() { edlib_core_leave(); }
};
