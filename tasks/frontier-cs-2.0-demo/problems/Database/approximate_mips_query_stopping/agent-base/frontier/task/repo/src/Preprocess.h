#pragma once
#include "StructType.h"
#include <cmath>
#include <assert.h>
#include <string>
#include <vector>
class Preprocess
{
public:
	Data data;
	float* SquareLen;
	Ben benchmark;
	float MaxLen;
	std::string data_file;
	std::string ben_file;
public:
	Preprocess(const std::string& path, const std::string& ben_file_);
	void load_data(const std::string& path);
	void cal_SquareLen();
	void ben_make();
	void ben_save();
	void ben_load();
	void ben_create();
	~Preprocess();
};

struct Dist_id
{
	int id;
	float dist;
	bool operator < (const Dist_id& rhs) {
		return dist < rhs.dist;
	}
};

class Partition
{
private:
	float ratio;
	void MakeChunks(Preprocess& prep);
public:
	int num_chunk;
	std::vector<float> MaxLen;
	//The chunk where each point belongs
	std::vector<int> chunks;
	//The data size of each chunks
	std::vector<int> nums;
	//The buckets by parti;
	std::vector<std::vector<int>> EachParti;
	std::vector<Dist_id> distpairs;
	void display();

	Partition(float c_, Preprocess& prep);

	Partition(float c_, float c0_, Preprocess& prep);
	//Partition() {}
	~Partition();
};

class PartitionNR
{
private:
	float ratio;
	void MakeChunks(Preprocess& prep);
public:
	int num_chunk;
	std::vector<float> MaxLen;
	//The chunk where each point belongs
	std::vector<int> chunks;
	//The data size of each chunks
	std::vector<int> nums;
	//The buckets by parti;
	std::vector<std::vector<int>> EachParti;

	void display();
	PartitionNR(int m_, Preprocess& prep);
	PartitionNR():num_chunk(0),ratio(0.0f) {}
	~PartitionNR();
};

class Parameter //N,dim,S, L, K, M, W;
{
public:
	int N;
	int dim;
	// Number of hash functions
	int S;
	//#L Tables; 
	int L;
	// Dimension of the hash table
	int K;
	//
	int MaxSize;
	//
	int KeyLen;

	int M = 1;

	int W = 0;

	float U;
	Parameter(Preprocess& prep, int L_, int K_, int M);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_);
	Parameter(Preprocess& prep, int L_, int K_, int M_, float U_, float W_);
	Parameter(Preprocess& prep, float c_, float S0);
	Parameter(Preprocess& prep, float c0_);
	bool operator = (const Parameter& rhs);
	~Parameter();
};


