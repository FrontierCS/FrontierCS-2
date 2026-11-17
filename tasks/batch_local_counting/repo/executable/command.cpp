//
// Created by Qiyan LI on 2022/8/30.
//

#include "command.h"

Command::Command(int argc, char **argv) : CommandParser(argc, argv){
    optionsKey[OptionKeyword::QueryGraphPath] = "-q";
    optionsKey[OptionKeyword::DataGraphPath] = "-d";
    optionsKey[OptionKeyword::BatchQuery] = "-b";
    optionsKey[OptionKeyword::ResultPath] = "-r";
    optionsKey[OptionKeyword::OptionalBatch] = "-share";
    optionsKey[OptionKeyword::TrianglePath] = "-t";
    booleanOptionValue[OptionKeyword::BatchQuery] = false;
    booleanOptionValue[OptionKeyword::OptionalBatch] = false;
    processOptions();
}

void Command::processOptions() {
    optionsValue[OptionKeyword::QueryGraphPath] = getCommandOption(optionsKey[OptionKeyword::QueryGraphPath]);
    optionsValue[OptionKeyword::DataGraphPath] = getCommandOption(optionsKey[OptionKeyword::DataGraphPath]);
    optionsValue[OptionKeyword::TrianglePath] = getCommandOption(optionsKey[OptionKeyword::TrianglePath]);
    optionsValue[OptionKeyword::ResultPath] = getCommandOption(optionsKey[OptionKeyword::ResultPath]);
    booleanOptionValue[OptionKeyword::BatchQuery] = commandOptionExists(optionsKey[OptionKeyword::BatchQuery]);
    booleanOptionValue[OptionKeyword::OptionalBatch] = commandOptionExists(optionsKey[OptionKeyword::OptionalBatch]);
}
