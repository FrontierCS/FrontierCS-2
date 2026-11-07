#pragma once
#include <set>
#include <algorithm>
#include "Preprocess.h"
#include <iterator>
#include <iostream>
template <class Query>
class Performance
{
public:
	//cost
	int cost;
	//
	std::vector<int> costs;
	// times of query
	int num;
	//
	float time_total;
	//
	float time_hash;
	//
	float time_sift;
	//
	float time_verify;
	//number of exact NN
	int NN_num;
	//number of results
	int res_num;
	//
	float ratio;
	// k-th ratio
	float kRatio=0.0f;
public:
	Performance();
	//update the Performance for each query
	void update(Query& query, Preprocess& prep);
	~Performance();
};

template <class Query>
Performance<Query>::Performance()
{
	cost = 0;
	num = 0;
	time_total = 0;
	time_hash = 0;
	time_sift = 0;
	time_verify = 0;
	NN_num = 0;
	res_num = 0;
	ratio = 0;
};

template <class Query>
void Performance<Query>::update(Query& query, Preprocess& prep)
{
	num++;
	cost += query.cost;
	time_sift += query.time_sift;
	time_verify += query.time_verify;
	time_total += query.time_total;

	int len = query.costs.size();
	if (costs.size() == 0)
		costs.resize(len);

	for (int i = 0; i < len; i++)
	{
		costs[i] += query.costs[i];
	}

	int num0 = query.res.size();
	if (num0 > query.k)
		num0 = query.k;
	res_num += num0;

	std::set<int> set1, set2;
	std::vector<int> set_intersection;
	set_intersection.clear();
	set1.clear();
	set2.clear();

	for (int j = 0; j < num0; j++)
	{
		float rate = query.res[j].inp / prep.benchmark.innerproduct[query.flag][j];
		ratio += rate;

		if(j==num0-1) kRatio+=rate;

		set1.insert(query.res[j].id);
		set2.insert((int)prep.benchmark.indice[query.flag][j]);
	}
	std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
		std::back_inserter(set_intersection));

	NN_num += set_intersection.size();
}

template <class Query>
Performance<Query>::~Performance()
{

}
