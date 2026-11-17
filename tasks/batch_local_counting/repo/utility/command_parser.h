//
// Created by anonymous author on 2022/7/20.
//

#ifndef SCOPE_COMMAND_PARSER_H
#define SCOPE_COMMAND_PARSER_H


#include <string>
#include <algorithm>
#include <vector>
class CommandParser {
private:
    std::vector<std::string> tokens_;

public:
    CommandParser(int argc, char **argv);
    std::string getCommandOption(const std::string &option) const;
    bool commandOptionExists(const std::string &option) const;
    int commandOptionType(const std::string &option) const;
};


#endif //SCOPE_COMMAND_PARSER_H
