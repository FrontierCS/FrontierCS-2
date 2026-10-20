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
