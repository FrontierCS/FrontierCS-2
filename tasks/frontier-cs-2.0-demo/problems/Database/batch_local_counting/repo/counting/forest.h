#ifndef SCOPE_FOREST_H
#define SCOPE_FOREST_H

#include "equation.h"

struct BatchState {
    int count;
    int orbitType;
    std::vector<Tree> trees;
    std::vector<Pattern> patterns;

    BatchState() : count(0), orbitType(0) {}

    void append(std::map<int, std::vector<Pattern>> &inputPatterns,
                std::map<int, std::vector<std::vector<Tree>>> &inputTrees);
};

#endif //SCOPE_FOREST_H
