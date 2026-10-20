//
// Created by anonymous author on 2022/7/20.
//

#include "command_parser.h"

CommandParser::CommandParser(const int argc, char **argv) {
    for (int i = 1; i < argc; ++i)
        tokens_.emplace_back(argv[i]);
}

std::string CommandParser::getCommandOption(const std::string &option) const {

    std::vector<std::string>::const_iterator itr;
    itr = find(tokens_.begin(), tokens_.end(), option);
    if (itr != tokens_.end() && ++itr != tokens_.end()) {
        return *itr;
    }
    return "";
}

bool CommandParser::commandOptionExists(const std::string &option) const {
    return find(tokens_.begin(), tokens_.end(), option) != tokens_.end();
}

int CommandParser::commandOptionType(const std::string &option) const {
    std::vector<std::string>::const_iterator itr;
    itr = find(tokens_.begin(), tokens_.end(), option);
    if (itr != tokens_.end() && ++itr != tokens_.end()) {
        return std::atoi(itr->c_str());
    }
    return 0;
}
