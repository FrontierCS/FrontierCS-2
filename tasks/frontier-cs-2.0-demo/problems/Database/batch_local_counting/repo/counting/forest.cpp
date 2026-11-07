#include "forest.h"

void BatchState::append(std::map<int, std::vector<Pattern>> &inputPatterns,
                        std::map<int, std::vector<std::vector<Tree>>> &inputTrees) {
    for (auto it = inputPatterns.begin(); it != inputPatterns.end(); ++it) {
        int factor = it->first;
        for (int i = 0; i < it->second.size(); ++i) {
            patterns.push_back(inputPatterns[factor][i]);
            trees.push_back(inputTrees[factor][i][0]);
            if (trees.size() == 1) orbitType = trees.back().getOrbitType();
            ++count;
        }
    }
}
