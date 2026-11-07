#include "mf_alsh.h"
#include "Preprocess.h"
#include "basis.h"
#include "query_core.h"
#include <memory>
#include <stdexcept>
#include <fstream>
#include <assert.h>
#include <random>
#include <iostream>
#include <fstream>
#include <map>
#include <ctime>
#include <sstream>
#include <numeric>
#include <iomanip>

#define MAXSIZE 20480
#define pi 3.141592653
#define CANDIDATES 100
#define MINFLOAT -3.40282e+038

using namespace mf_alsh;

Hash::Hash(Preprocess& prep_, Parameter& param_,
	const std::string& file, Partition& part_, const std::string& funtable)
	:parti(part_)
{
	N = param_.N;
	dim = param_.dim;
	L = param_.L;
	K = param_.K;
	S = param_.S;

	policy_resource = funtable;

	std::cout << std::endl << "START HASHING..." << std::endl << std::endl;
	lsh::timer timer;

	std::cout << "SETTING HASH PARAMETER..." << std::endl;
	timer.restart();
	SetHash();
	std::cout << "SETTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	std::cout << "COMPUTING HASH..." << std::endl;
	timer.restart();
	GetHash(prep_);
	std::cout << "COMPUTING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;

	std::cout << "BUILDING INDEX..." << std::endl;
	std::cout << "THERE ARE " << L << " " << K << "-D HASH TABLES." << std::endl;
	timer.restart();
	GetTables(prep_);
	std::cout << "BUILDING TIME: " << timer.elapsed() << "s." << std::endl << std::endl;


}

bool Hash::IsBuilt(const std::string& file)
{
	int file_md = L * 1000 + K;
	std::string res;
	std::stringstream ss;
	ss << file_md;
	ss >> res;
	std::string index_suf;
	index_suf.clear();
	index_suf = '_';
	index_suf.append(res);

	index_file = file;
	index_file.append(index_suf);

	std::ifstream in(index_file, std::ios::binary);
	return in.good();
}

void Hash::SetHash()
{
	hashpar.rndAs1 = new float* [S];
	hashpar.rndAs2 = new float* [S];


	for (int i = 0; i < S; i++){
		hashpar.rndAs1[i] = new float[dim];
		hashpar.rndAs2[i] = new float[1];
	}

	std::mt19937 rng(int(std::time(0)));
	std::normal_distribution<float> nd;
	for (int j = 0; j < S; j++)
	{
		for (int i = 0; i < dim; i++)
		{
			hashpar.rndAs1[j][i] = (nd(rng));
		}
		for (int i = 0; i < 1; i++)
		{
			hashpar.rndAs2[j][i] = (nd(rng));
		}
	}
}

void Hash::GetHash(Preprocess& prep)
{
	float* dataExpend = new float[N];
	std::mt19937 rng(int(std::time(0)));
	std::uniform_real_distribution<float> ur(-1, 1);
	int count = 0;
	for (int j = 0; j < N; j++)
	{
		assert(parti.MaxLen[parti.chunks[j]] >= prep.SquareLen[j]);
		dataExpend[j] = sqrt(parti.MaxLen[parti.chunks[j]] - prep.SquareLen[j]);
		if (ur(rng) > 0) {
			dataExpend[j] *= -1;
			++count;
		}

	}
	std::cout << "RXT ratio: " << (double)count / N << std::endl;

	hashval = new float* [N];
	for (int j = 0; j < N; j++) {
		hashval[j] = new float[S];
		for (int i = 0; i < S; i++) {
			hashval[j][i] = cal_inner_product(prep.data.val[j], hashpar.rndAs1[i], dim) + dataExpend[j] * hashpar.rndAs2[i][0];
		}
	}
	delete[] dataExpend;
}

void Hash::GetTables(Preprocess& prep)
{
	int i, j, k;

	int num_bucket = 1 << K;

	myIndexes = new std::vector<int> * *[parti.num_chunk];
	for (j = 0; j < parti.num_chunk; ++j) {
		myIndexes[j] = new std::vector<int> * [L];
		for (i = 0; i < L; ++i) {
			myIndexes[j][i] = new std::vector<int>[num_bucket];
		}
	}

	for (j = 0; j < L; j++) {
		for (i = 0; i < N; i++) {
			int start = j * K;
			int key = 0;
			for (k = 0; k < K; k++) {
				key = key << 1;
				if (this->hashval[i][ start + k] > 0) {
					++key;
				}
			}
			myIndexes[(size_t)parti.chunks[i]][j][key].push_back(i);
		}
	}
}

Hash::~Hash()
{
	printf("Destroy Index!\n");
	for (int j = 0; j < parti.num_chunk; ++j) {
		for (int i = 0; i < L; ++i) {
			for (int l = 0; l < (1 << K); ++l) {
				std::vector<int>().swap(myIndexes[j][i][l]);
			}
			delete[] myIndexes[j][i];
		}
		delete[] myIndexes[j];
	}
	delete[] myIndexes;

	clear_2d_array(hashval, N);
	clear_2d_array(hashpar.rndAs1, S);
	clear_2d_array(hashpar.rndAs2, S);
}


Query::Query(int id,float c_,int k_,Hash& hash,Preprocess& prep,int ub_) {
    flag=id;c=c_;k=k_;UB=k_;dim=prep.data.dim;mydata=prep.data.val;
    query_point=mydata[id];norm=std::sqrt(prep.SquareLen[id]);
    if(!(norm>0)||k<1||k>100) throw std::runtime_error("query inputs");
    lsh::timer timer;
    std::vector<double> projections(hash.S);
    for(int i=0;i<hash.S;++i)
        projections[i]=cal_inner_product(query_point,hash.hashpar.rndAs1[i],dim)/norm;
    auto core=make_query_core();
    core->reset(hash.L,hash.K,projections.data());
    time_hash=timer.elapsed(); timer.restart();
    // A query-global sequence is cached and replayed for smaller norm partitions.
    std::vector<Probe> sequence;
    std::vector<bool> emitted(hash.L*(1<<hash.K),false),seen(hash.N,false);
    std::vector<Res> best;
    costs.assign(hash.parti.num_chunk,0);
    auto verify=[&](int x,int t) {
        if(seen[x]) return;
        seen[x]=true;++costs[t];++cost;
        Res r(x,cal_inner_product(mydata[x],query_point,dim));
        if(best.size()<static_cast<size_t>(k)) {best.push_back(r);std::push_heap(best.begin(),best.end());}
        else if(r.inp>best.front().inp) {
            std::pop_heap(best.begin(),best.end());best.back()=r;std::push_heap(best.begin(),best.end());
        }
    };
    for(int t=hash.parti.num_chunk-1;t>=0;--t) {
        double maximum=norm*std::sqrt(hash.parti.MaxLen[t]);
        if(best.size()==static_cast<size_t>(k)&&best.front().inp>=c*maximum) break;
        // Author optimization: small partitions use direct verification.
        if(hash.parti.nums[t]<400) {for(int x:hash.parti.EachParti[t]) verify(x,t);continue;}
        size_t cursor=0;
        while(costs[t]<hash.parti.nums[t]) {
            if(cursor==sequence.size()) {
                Probe p;
                if(!core->next(p)) break;
                if(p.table<0||p.table>=hash.L||p.key<0||p.key>=(1<<hash.K)||!std::isfinite(p.score))
                    throw std::runtime_error("invalid probe");
                int code=p.table*(1<<hash.K)+p.key;
                if(emitted[code]) throw std::runtime_error("duplicate probe");
                emitted[code]=true;
                double authentic=0;
                for(int j=0;j<hash.K;++j) {
                    double z=projections[p.table*hash.K+j];
                    if(((p.key>>(hash.K-1-j))&1)!=(z>0)) authentic+=z*z;
                }
                if(std::abs(p.score-authentic)>1e-8*(1+authentic))
                    throw std::runtime_error("false quantization distance");
                sequence.push_back(p);
            }
            const Probe& p=sequence[cursor];
            SearchState state={static_cast<int>(best.size()),k,costs[t],hash.parti.nums[t],
                best.size()==static_cast<size_t>(k)?best.front().inp:0,p.score,c,maximum,
                hash.L,hash.policy_resource.c_str()};
            if(core->stop(state)) break;
            for(int x:hash.myIndexes[t][p.table][p.key]) verify(x,t);
            ++cursor;
        }
    }
    res=best;
    std::sort(res.begin(),res.end(),[](const Res&a,const Res&b){return a.inp>b.inp;});
    time_sift=timer.elapsed();time_verify=time_sift;time_total=time_hash+time_sift;
}
Query::~Query() {}
