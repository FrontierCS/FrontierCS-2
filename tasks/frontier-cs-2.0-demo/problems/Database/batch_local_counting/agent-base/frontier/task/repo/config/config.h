//
// Created by anonymous author on 2022/7/20.
//

#ifndef SCOPE_CONFIG_H
#define SCOPE_CONFIG_H

#include <cstdlib>
#include <utility>
#include <cstdint>

/**
 * Set intersection method.
 * 0: Hybrid method; 1: Merge based set intersections.
 */
#define HYBRID 0

/**
 * Accelerate set intersection with SIMD instructions.
 * 0: AVX2; 1: AVX512; 2: Basic;
 * If building in mac AMD chip, can only use 2.
 * For machines using Intel CPU, use 0.
 */

#ifdef __APPLE__
#define SI 2
#else
#define SI 0
#endif

#define MAX_PATTERN_SIZE 10
#define MAX_NUM_NODE 10
//#define COLLECT_STATISTICS
//#define ONLY_PLAN
typedef uint32_t ui;
typedef uint32_t VertexID;
typedef uint32_t EdgeID;
typedef int64_t Count;
typedef Count *HashTable;
typedef uint64_t CanonType;
typedef std::pair<VertexID, VertexID> Edge;
//#define DEBUG

#endif //SCOPE_CONFIG_H
