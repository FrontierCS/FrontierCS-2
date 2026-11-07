// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

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

#include "Preprocess.h"
#include "mf_alsh.h"
#include "basis.h"
#include "alg.h"

extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;

// Fargo Project: Revised by Xi ZHAO -- Nov 16, 2022

// For PVDLB 2023: FARGO: Fast Maximum Inner Product Search via Global Multi-Probing

// For any question, please feel free to contact me. Email: xzhaoca@connect.ust.hk

int main(int argc, char const* argv[])
{
	std::string dataset = "gist";
	if (argc > 1) {
		dataset = argv[1];
	}
	std::string argvStr[4];
	argvStr[1] = (dataset + ".data");
	argvStr[2] = (dataset + ".index");
	argvStr[3] = (dataset + ".ben");

	float c = 0.9f;
	int k = 50;
	int m, L, K;

	std::cout << "Using FARGO for " << argvStr[1] << std::endl;
	Preprocess prep(data_fold1 + (argvStr[1]), data_fold2 + (argvStr[3]));
	std::vector<resOutput> res;
	m = 0;
	L = 5;
	K = 12;
	c = 0.8;

	Parameter param(prep, L, K, 1);

	lsh::timer timer;
	Partition parti(c, prep);
	mf_alsh::Hash myslsh(prep, param, index_fold.append(argvStr[2]), parti, "");

	std::vector<int> ms = { 0,100,200,300,400,500,600,700,800,900,1000};
	for (auto& x : ms) {
		m = x + k;
		res.push_back(Alg0_mfalsh(myslsh, c, m, k, L, K, prep));
	}

	saveAndShow(c, k, dataset, res);

	return 0;
}
