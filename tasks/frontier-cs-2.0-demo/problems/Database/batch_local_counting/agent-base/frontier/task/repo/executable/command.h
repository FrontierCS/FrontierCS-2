//
// Created by anonymous author on 2022/8/30.
//

#ifndef SCOPE_COMMAND_H
#define SCOPE_COMMAND_H

#include "command_parser.h"
#include <iostream>
#include <map>

enum OptionKeyword {
    QueryGraphPath = 1,     // -q, the query graph file path
    DataGraphPath = 2,      // -d, the data graph file path
    TrianglePath = 3,       // -t, the triangle binary file path
    ResultPath = 4,         // -r, the result file path
    BatchQuery = 5,         // -b, batch query or single query
    OptionalBatch = 6,
};

class Command : public CommandParser {
private:
    std::map<OptionKeyword, std::string> optionsKey;
    std::map<OptionKeyword, std::string> optionsValue;
    std::map<OptionKeyword, bool> booleanOptionValue;
    std::map<OptionKeyword, int> intOptionValue;

private:
    void processOptions();

public:
    Command(int argc, char **argv);

    std::string getQueryGraphPath() {
        return optionsValue[OptionKeyword::QueryGraphPath];
    }

    std::string getDataGraphPath() {
        return optionsValue[OptionKeyword::DataGraphPath];
    }

    std::string getTrianglePath() {
        return optionsValue[OptionKeyword::TrianglePath];
    }

    std::string getResultPath() {
        return optionsValue[OptionKeyword::ResultPath];
    }

    bool getBatchQuery() {
        return booleanOptionValue[OptionKeyword::BatchQuery];
    }

};


#endif //SCOPE_COMMAND_H
