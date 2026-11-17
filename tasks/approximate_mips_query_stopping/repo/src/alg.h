#pragma once
#include <string>
#include "Preprocess.h"
#include "mf_alsh.h"
#include "performance.h"
#include "basis.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>

struct resOutput
{
	std::string algName;
	int L;
	int K;
	float c;
	float time;
	float recall;
	float ratio;
	float cost;
	float kRatio;
};

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

// Return true when the current query should stop consuming the global probe
// frontier.  The starter policy is the fixed-candidate-budget baseline.
bool should_stop(const SearchState& search_state);

#if defined(unix) || defined(__unix__)
inline void localtime_s(tm* ltm, time_t* now) {}
#endif

inline resOutput Alg0_mfalsh(mf_alsh::Hash& myslsh, float c_, int m_, int k_, int L_, int K_, Preprocess& prep)
{
	std::string query_result = ("results/MF_ALSH_result.csv");

	lsh::timer timer;
	std::cout << std::endl << "RUNNING QUERY ..." << std::endl;

	int Qnum = 100;
	lsh::progress_display pd(Qnum);
	Performance<mf_alsh::Query> perform;
	lsh::timer timer1;
	for (int j = 0; j < Qnum; j++)
	{
		mf_alsh::Query query(j, c_, k_, myslsh, prep, m_);
		perform.update(query, prep);
		++pd;
	}

	float mean_time = (float)perform.time_total / perform.num;
	std::cout << "AVG QUERY TIME:    " << mean_time * 1000 << "ms." << std::endl << std::endl;
	std::cout << "AVG RECALL:        " << ((float)perform.NN_num) / (perform.num * k_) << std::endl;
	std::cout << "AVG RATIO:         " << ((float)perform.ratio) / (perform.res_num) << std::endl;

	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);

	resOutput res;
	res.algName = "FARGO";
	res.L = myslsh.L;
	res.K = myslsh.K;
	res.c=c_;
	res.time = mean_time * 1000;
	res.recall = ((float)perform.NN_num) / (perform.num * k_);
	res.ratio = ((float)perform.ratio) / (perform.res_num);
	res.cost = ((float)perform.cost) / ((long long)perform.num * (long long)myslsh.N);
	res.kRatio=perform.kRatio/perform.num;
	//delete[] ltm;
	return res;
}

void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res);
