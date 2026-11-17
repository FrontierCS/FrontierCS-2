#include <string>
#include <cstdint>
#include <span>
#include <optional>

namespace spectrum {
    std::optional<std::basic_string<uint8_t>> from_hex(std::string_view hex) noexcept;
    std::string to_hex(std::span<uint8_t> bytes) noexcept;
} // namespace spectrum
