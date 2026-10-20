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


    class Query {
        float c; int UB,dim;float** mydata;float* query_point;
    public:
        int k,flag,cost=0;float norm;
        std::vector<int> costs;
        float time_total=0,time_hash=0,time_sift=0,time_verify=0;
        std::vector<Res> res;
        Query(int id,float c,int k,Hash& hash,Preprocess& prep,int capacity);
        ~Query();
    };
}
