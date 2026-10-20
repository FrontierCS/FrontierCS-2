//
// Created by anonymous author on 2022/11/21.
//

#include "utils.h"

int quickPow10(int n)
{
    static int pow10[10] = {
            1, 10, 100, 1000, 10000,
            100000, 1000000, 10000000, 100000000, 1000000000
    };

    return pow10[n];
}

Count choosec(Count n, int k)
{
    if (n < 0)
        return 0;
    if (k > n)
        return 0;
    if (k == 0)
        return 1;
    Count nom = n;
    Count denom = 1;
    for (int i = 1; i < k; i++)
    {
        nom *= (n-i);
        denom *= (i+1);
    }
    return nom / denom;
}

void leftShit(std::vector<bool> &arr, int k) {
    int count = 0;
    for (int j = k; j >= 0; --j)
        if ( arr[j] == 1)  ++count;
    if (count == 0)  return;
    for (int j = k; j >= 0; --j)
        arr[j] = false;
    for ( int j=0; j<count;++j)
        arr[j] = true;
}

std::vector<std::vector<bool>> chooseK(ui n, int k) {
    if (k > n) {
        std::vector<std::vector<bool>> result;
        result.emplace_back(n, true);
        return result;
    }
    std::vector<bool> arr(n, false);
    for (int i = 0; i < k; ++i)
        arr[i] = true;
    std::vector<std::vector<bool>> result;
    result.push_back(arr);
    bool flag = true;
    while (flag) {
        flag = false;
        for (int i = 0; i < n - 1; ++i) {
            if (arr[i] && !arr[i + 1]){
                flag = true;
                arr[i] = false;
                arr[i + 1] = true;
                leftShit(arr,i);
                result.push_back(arr);
                break;
            }
        }
    }

    return result;
}

void sortEOrbitAggrePos(std::vector<int> &aggrePos) {
    std::vector<std::pair<int, int>> tmp(aggrePos.size() / 2);
    for (int i = 0; i < aggrePos.size(); i = i + 2) {
        if (aggrePos[i] < aggrePos[i + 1])
            tmp[i / 2] = std::make_pair(aggrePos[i], aggrePos[i + 1]);
        else
            tmp[i / 2] = std::make_pair(aggrePos[i + 1], aggrePos[i]);
    }
    std::sort(tmp.begin(), tmp.end());
    for (int i = 0; i < aggrePos.size(); i = i + 2) {
        aggrePos[i] = tmp[i / 2].first;
        aggrePos[i + 1] = tmp[i / 2].second;
    }
}

ui firstPosGreaterThan(const ui *candidate, ui begin, ui end, ui target) {
    int low = (int)begin;
    int high = (int)end - 1;
    int mid;
    while (low <= high) {
        mid = low + ((high - low) >> 1);
#ifndef __APPLE__
        _mm_prefetch((char *) &candidate[(mid + 1 + high) / 2], _MM_HINT_T0);
        _mm_prefetch((char *) &candidate[(mid - 1 + low) / 2], _MM_HINT_T0);
#endif
        if (candidate[mid] == target) {
            return mid + 1;
        } else if (candidate[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    return (ui)low;
}

// Backtracking function to generate all permutations
void generatePermutations(ui n, ui** automorphisms, ui* current, bool* used, size_t& index, size_t currentSize) {
    if (currentSize == n) {
        for (ui j = 0; j < n; j++) {
            automorphisms[index][j] = current[j];
        }
        index++;
        return;
    }

    for (ui i = 0; i < n; i++) {
        if (!used[i]) {
            used[i] = true;
            current[currentSize] = i;
            generatePermutations(n, automorphisms, current, used, index, currentSize + 1);
            used[i] = false;
        }
    }
}

void generateCliqueRulesHelper(const std::vector<VertexID> &vertices,
                               std::vector<std::vector<std::vector<VertexID>>> &rules,
                               std::vector<std::vector<VertexID>> current) {
    if (vertices.size() == 2) {
        current.push_back(vertices);
        rules.push_back(current);
        return;
    }
    for (int i = 0; i < vertices.size(); ++i) {
        std::vector<VertexID> newVertices;
        std::vector<VertexID> verticesRule;
        newVertices.reserve(vertices.size());
        verticesRule.reserve(vertices.size());
        verticesRule.push_back(vertices[i]);
        for (int j = 0; j < vertices.size(); ++j) {
            if (j != i) {
                newVertices.push_back(vertices[j]);
                verticesRule.push_back(vertices[j]);
            }
        }
        current.push_back(verticesRule);
        generateCliqueRulesHelper(newVertices, rules, current);
        current.pop_back();
    }
}

std::vector<std::vector<std::vector<VertexID>>> generateCliqueRules(const std::vector<VertexID> &vertices,
                                                                    const std::vector<std::vector<std::vector<VertexID>>> &candidateRules) {
    if (vertices.size() < 2) return candidateRules;
    std::vector<std::vector<VertexID>> current;
    std::vector<std::vector<std::vector<VertexID>>> rules;
    std::vector<std::vector<std::vector<VertexID>>> newCandRules;
    generateCliqueRulesHelper(vertices, rules, current);
    if (candidateRules.empty()) newCandRules = rules;
    else {
        for (int i = 0; i < candidateRules.size(); ++i) {
            for (int j = 0; j < rules.size(); ++j) {
                std::vector<std::vector<VertexID>> newRule = candidateRules[i];
                for (const auto &rule : rules[j])
                    newRule.push_back(rule);
                newCandRules.push_back(newRule);
            }
        }
    }

    return newCandRules;
}

void write2DVectorPairToStream(std::ofstream& outFile, const std::vector<std::vector<std::pair<int, int>>>& vec) {
    size_t outerSize = vec.size();
    outFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

    for (const auto& innerVec : vec) {
        size_t innerSize = innerVec.size();
        outFile.write(reinterpret_cast<const char*>(&innerSize), sizeof(innerSize));

        for (const auto& pair : innerVec) {
            outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
            outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
        }
    }
}

void read2DVectorPairFromStream(std::ifstream& inFile, std::vector<std::vector<std::pair<int, int>>>& vec) {
    size_t outerSize;
    inFile.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    vec.resize(outerSize);

    for (auto& innerVec : vec) {
        size_t innerSize;
        inFile.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        innerVec.resize(innerSize);

        for (auto& pair : innerVec) {
            inFile.read(reinterpret_cast<char*>(&pair.first), sizeof(pair.first));
            inFile.read(reinterpret_cast<char*>(&pair.second), sizeof(pair.second));
        }
    }
}

void writeVectorPairToStream(std::ofstream& outFile, const std::vector<std::pair<int, int>>& vec) {
    size_t size = vec.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));

    for (const auto& pair : vec) {
        outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
        outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
    }
}

void readVectorPairFromStream(std::ifstream& inFile, std::vector<std::pair<int, int>>& vec) {
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);

    for (auto& pair : vec) {
        inFile.read(reinterpret_cast<char*>(&pair.first), sizeof(pair.first));
        inFile.read(reinterpret_cast<char*>(&pair.second), sizeof(pair.second));
    }
}

std::string pathJoin(const std::string& part1, const std::string& part2) {
    if (part1.empty()) {
        return part2;
    }
    if (part2.empty()) {
        return part1;
    }

    // Define the directory separator
#ifdef _WIN32
    const char sep = '\\';
#else
    const char sep = '/';
#endif

    std::string result = part1;

    // Add separator if needed
    if (result.back() != sep) {
        result += sep;
    }

    // Append the second part, avoiding double separator if it starts with one
    if (part2.front() == sep) {
        result += part2.substr(1);
    } else {
        result += part2;
    }

    return result;
}
