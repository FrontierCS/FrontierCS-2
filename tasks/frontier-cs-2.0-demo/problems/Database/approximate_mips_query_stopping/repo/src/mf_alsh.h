#pragma once
#include "StructType.h"
#include "Preprocess.h"
#include <cmath>
#include <assert.h>
#include <vector>
#include <queue>
#include <cfloat>

namespace mf_alsh
{
	class Hash
	{
	private:
		std::string index_file;
		
	public:
		int N;
		int dim;
		// Number of hash functions
		int S;
		//#L Tables; 
		int L;
		// Dimension of the hash table
		int K;

		float** hashval;
		Partition parti;
		HashParam hashpar;
		std::vector<int>*** myIndexes;

		std::string policy_resource;
	public:
		Hash(Preprocess& prep_, Parameter& param_, const std::string& file, Partition& part_, const std::string& funtable);
		void SetHash();
		void GetHash(Preprocess& prep);
		void GetTables(Preprocess& prep);
		bool IsBuilt(const std::string& file);
		~Hash();
	};

	struct Res//the result of knns
	{
		float inp;
		int id;
		Res() = default;
		Res(int id_, float inp_):id(id_), inp(inp_) {}
		bool operator < (const Res& rhs) const {
			return inp > rhs.inp;
		}
	};

	struct hash_pair
	{
		float val;
		int bias;

		bool operator < (const hash_pair& rhs) const {
			return val < rhs.val;
		}
	};

	struct indice_pair
	{
		int key;
		int end;
		int table_id;
		float score;
		bool operator < (const indice_pair& rhs) const {
			//return score > rhs.score;
			return score > rhs.score
				//|| (score == rhs.score && key > rhs.key)
				//|| (score == rhs.score && key == rhs.key && table_id > rhs.table_id)
				;
		}
	};

	struct mp_pair
	{
		//int key;
		int end;
		//int table_id;
		float score = FLT_MAX;
	};

	struct gmp_pair
	{
		int key;
		//int end;
		int table_id;
		float score;
		bool operator < (const gmp_pair& rhs) const {
			//return score > rhs.score;
			return score < rhs.score
				//|| (score == rhs.score && key > rhs.key)
				//|| (score == rhs.score && key == rhs.key && table_id > rhs.table_id)
				;
		}

		//bool operator > (const gmp_pair& rhs) const {
		//	//return score > rhs.score;
		//	return score < rhs.score
		//		//|| (score == rhs.score && key > rhs.key)
		//		//|| (score == rhs.score && key == rhs.key && table_id > rhs.table_id)
		//		;
		//}
	};

	class Query
	{
	private:
		// the parameter "c" in "c-ANN"
		float c;
		//which chunk is accessed
		int chunks;
		int UB = 100;
		float* query_point;

		// the hash value of query point
		float* hashval;

		std::vector<std::vector<hash_pair>> weigh;
		std::vector<float> total_score;

		float** mydata;
		int dim;

		float inp_LB;
		// Set of points sifted
		//std::vector<int> candidate;

		std::vector<int> keys;

		void shift(indice_pair& ip0, indice_pair& res);
		void expand(indice_pair& ip0, indice_pair& res);

		std::priority_queue<indice_pair> global_min;
		//gmp_pair gmp_buckets[20480];
		//mp_pair gb[5][4096];
		indice_pair* ProbingSequence;
		int SequenceLen;

		int tid = -1;

		//void expand(indice_pair& ip0, indice_pair& res)
		//{
		//	res = ip0;
		//	++res.end;
		//	res.key += weigh[res.table_id][res.end].bias;
		//	res.score = ip0.score + weigh[res.table_id][res.end].val;
		//}

		//void gmp_generator(int id)
		//{
		//	//int& bid = gb[id_].key;
		//	int end = gb[tid][id].end + 1;
		//	if (end < 12) {
		//		//shift
		//		int bs = id + weigh[tid][end].bias - weigh[tid][end - 1].bias;
		//		gb[tid][bs].end = end;
		//		gb[tid][bs].score = gb[tid][id].score + weigh[tid][end].val - weigh[tid][end - 1].val;
		//		gmp_generator(bs);

		//		//expand
		//		int be = id + weigh[tid][end].bias;
		//		gb[tid][be].end = end;
		//		gb[tid][be].score = gb[tid][id].score + weigh[tid][end].val;
		//		gmp_generator(be);
		//	}
		//	
		//}

	public:
		// k-NN
		int k;
		// Indice of query point in dataset. Be equal to -1 if the query point isn't in the dataset.
		int flag;
		//
		float norm;
		//
		int cost = 0;
		//cost of each partition
		std::vector<int> costs;
		//
		float time_total = 0;
		//
		float time_hash = 0;
		//
		float time_sift = 0;

		float time_verify = 0;
		// query result:<indice of ANN,distance of ANN>
		std::vector<Res> res;

		void cal_hash(Hash& hash, Preprocess& prep);
		void sift(Hash& hash, Preprocess& prep);
		void knn(std::priority_queue<Res>& res_PQ,
			Hash& hash, Preprocess& prep,
			std::vector<int>** table,
			std::vector<bool>& flag, int& num_res);
		void siftF(Hash& hash, Preprocess& prep);
		void knnF(Res* res_PQ, Hash& hash, Preprocess& prep, std::vector<int>** table, std::vector<bool>& flag_, int& size);
	public:
		Query(int id, float c_, int k_, Hash& hash, Preprocess& prep, int ub_);

		~Query();
	};
}

