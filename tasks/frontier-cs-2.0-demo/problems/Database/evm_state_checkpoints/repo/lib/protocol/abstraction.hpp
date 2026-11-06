#pragma once
#include <spectrum/common/statistics.hpp>

namespace spectrum {

class Protocol {

    public:
    virtual void Start() = 0;
    virtual void Stop() = 0;
    virtual ~Protocol() = default;

};

} // namespace spectrum