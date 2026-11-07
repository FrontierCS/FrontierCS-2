#include <fmt/core.h>
#include <iomanip>

void PrefixFormatter(std::ostream& s, const google::LogMessage& m, void* data) {
    auto color = [&m]{
        switch (m.severity()) {
            case 0: return "\e[1;36m";
            case 1: return "\e[1;33m";
            case 2: return "\e[1;31m";
            default: return "\e[0;30m";
        }
    }();
    auto align = [&](std::ostream& s , bool left, size_t x, std::string str) {
        s << std::setfill(' ');
        if (left) s << std::setiosflags(std::ios::left);
        s << std::setw((str.size() + x - 1) / x * x) << str;
    };
    align(s << color, true, 8, google::GetLogSeverityName(m.severity()));
    s << "\e[0;30m" << " | ";
    s << std::setw(15) << m.thread_id() << " | ";
    align(s, true, 16, fmt::format("{}:{}", m.basename(), m.line()));
    s << " |";
}
