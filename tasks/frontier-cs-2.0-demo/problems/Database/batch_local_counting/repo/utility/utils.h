//
// Created by anonymous author on 2022/11/21.
//

#ifndef SCOPE_UTILS_H
#define SCOPE_UTILS_H

#include "config.h"
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <type_traits>
#include <set>

#ifndef __APPLE__
#include <immintrin.h>
#include <x86intrin.h>
#endif

int quickPow10(int n);
Count choosec(Count n, int k);
void leftShit(std::vector<bool> &arr, int k);
std::vector<std::vector<bool>> chooseK(ui n, int k);
void sortEOrbitAggrePos(std::vector<int> &aggrePos);
ui firstPosGreaterThan(const ui *candidate, ui begin, ui end, ui target);
void generatePermutations(ui n, ui** automorphisms, ui* current, bool* used, size_t& index, size_t currentSize);
void generateCliqueRulesHelper(const std::vector<VertexID> &vertices,
                               std::vector<std::vector<std::vector<VertexID>>> &rules,
                               std::vector<std::vector<VertexID>> current);
std::vector<std::vector<std::vector<VertexID>>> generateCliqueRules(const std::vector<VertexID> &vertices,
                                                                    const std::vector<std::vector<std::vector<VertexID>>> &candidateRules);

template <typename T>
void writeVectorToStream(std::ofstream& outFile, const std::vector<T>& vec) {
    size_t size = vec.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if constexpr (std::is_same<T, std::string>::value) {
        for (const auto& item : vec) {
            size_t length = item.size();
            outFile.write(reinterpret_cast<const char*>(&length), sizeof(length));
            outFile.write(item.c_str(), length);
        }
    }
    else if constexpr (std::is_same<T, bool>::value) {
        // Special handling for std::vector<bool>
        for (bool item : vec) {
            char value = item ? 1 : 0;
            outFile.write(&value, sizeof(value));
        }
    } else {
        // Handling for all other types
        for (const auto& item : vec) {
            outFile.write(reinterpret_cast<const char*>(&item), sizeof(T));
        }
    }
}

template <typename T>
void readVectorFromStream(std::ifstream& inFile, std::vector<T>& vec) {
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    if constexpr (std::is_same<T, std::string>::value) {
        for (size_t i = 0; i < size; ++i) {
            size_t length;
            inFile.read(reinterpret_cast<char*>(&length), sizeof(length));
            std::string temp(length, '\0');
            inFile.read(&temp[0], length);
            vec[i] = temp;
        }
    }
    else if constexpr (std::is_same<T, bool>::value) {
        // Special handling for std::vector<bool>
        for (size_t i = 0; i < size; ++i) {
            char value;
            inFile.read(&value, sizeof(char));
            vec[i] = (value != 0);
        }
    } else {
        // Handling for all other types
        inFile.read(reinterpret_cast<char*>(vec.data()), sizeof(T) * size);
    }
}

template <typename T>
void writeArrayToStream(std::ofstream& outFile, const T* array, size_t size) {
    if (array != nullptr && size > 0) {
        outFile.write(reinterpret_cast<const char*>(array), sizeof(T) * size);
    }
}

template <typename T>
void readArrayFromStream(std::ifstream& inFile, T*& array, size_t size) {
    if (size > 0) {
        array = new T[size];
        inFile.read(reinterpret_cast<char*>(array), sizeof(T) * size);
    } else {
        array = nullptr;
    }
}

template <typename T>
void write2DArrayToStream(std::ofstream& outFile, T** array2D, size_t dim1Size, size_t dim2Size) {
    if (array2D != nullptr && dim1Size > 0) {
        for (size_t i = 0; i < dim1Size; ++i) {
            writeArrayToStream(outFile, array2D[i], dim2Size);
        }
    }
}

template <typename T>
void read2DArrayFromStream(std::ifstream& inFile, T**& array2D, size_t dim1Size, size_t dim2Size) {
    array2D = new T*[dim1Size];
    for (size_t i = 0; i < dim1Size; ++i) {
        readArrayFromStream(inFile, array2D[i], dim2Size);
    }
}

template <typename T>
void writeSetToStream(std::ofstream& outFile, const std::set<T>& set) {
    size_t size = set.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
    for (const T& item : set) {
        outFile.write(reinterpret_cast<const char*>(&item), sizeof(T));
    }
}

template <typename T>
void readSetFromStream(std::ifstream& inFile, std::set<T>& set) {
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    set.clear();
    T value;
    for (size_t i = 0; i < size; ++i) {
        inFile.read(reinterpret_cast<char*>(&value), sizeof(T));
        set.insert(value);
    }
}

template <typename T>
void write2DVectorToStream(std::ofstream& outFile, const std::vector<std::vector<T>>& vec) {
    size_t outerSize = vec.size();
    outFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

    for (const auto& innerVec : vec) {
        writeVectorToStream(outFile, innerVec);
    }
}

template <typename T>
void read2DVectorFromStream(std::ifstream& inFile, std::vector<std::vector<T>>& vec) {
    size_t outerSize;
    inFile.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    vec.resize(outerSize);

    for (auto& innerVec : vec) {
        readVectorFromStream(inFile, innerVec);
    }
}

template <typename T>
void write3DVectorToStream(std::ofstream& outFile, const std::vector<std::vector<std::vector<T>>>& vec) {
    size_t outerSize = vec.size();
    outFile.write(reinterpret_cast<const char*>(&outerSize), sizeof(outerSize));

    for (const auto& middleVec : vec) {
        write2DVectorToStream(outFile, middleVec);
    }
}

template <typename T>
void read3DVectorFromStream(std::ifstream& inFile, std::vector<std::vector<std::vector<T>>>& vec) {
    size_t outerSize;
    inFile.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    vec.resize(outerSize);

    for (auto& middleVec : vec) {
        read2DVectorFromStream(inFile, middleVec);
    }
}

void write2DVectorPairToStream(std::ofstream& outFile, const std::vector<std::vector<std::pair<int, int>>>& vec);
void read2DVectorPairFromStream(std::ifstream& inFile, std::vector<std::vector<std::pair<int, int>>>& vec);
void writeVectorPairToStream(std::ofstream& outFile, const std::vector<std::pair<int, int>>& vec);
void readVectorPairFromStream(std::ifstream& inFile, std::vector<std::pair<int, int>>& vec);
std::string pathJoin(const std::string& part1, const std::string& part2);

#endif //SCOPE_UTILS_H
