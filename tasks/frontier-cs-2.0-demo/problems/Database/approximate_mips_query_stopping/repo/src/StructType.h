#pragma once
#include <cstddef>

struct Data
{
	// Dimension of data
	int dim;
	// Number of data
	int N;
	// Data matrix
	float** val;
};

struct Ben
{
	int N;
	int num;
	int** indice;
	float** innerproduct;
};

struct HashParam
{
	// the value of a in S hash functions
	float** rndAs1;
	// the value of a in S hash functions
	float** rndAs2;
};

// Online state made available to the query stopping policy.  The score array
// is the current result heap; the frontier fields describe the globally
// ordered buckets that have not yet been consumed.
struct SearchState
{
	const float* top_scores;
	std::size_t top_score_count;
	float kth_score;
	float frontier_score;
	int probes_performed;
	int candidates_verified;
	int candidate_budget;
	int remaining_frontier;
	float approximation_ratio;
	float maximum_inner_product;
	int hash_tables;
	const char* policy_resource;
};
