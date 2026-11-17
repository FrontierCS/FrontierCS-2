#include "alg.h"

#if defined(unix) || defined(__unix__)
//std::string data_fold = "/home/xizhao/dataset/", index_fold = " ";
std::string data_fold = "./../../wd/dataset/", index_fold = " ";
std::string data_fold1 = data_fold, data_fold2 = data_fold+("MIPS/");
#else
std::string data_fold = "E:/Dataset_for_c/", index_fold = " ";
std::string data_fold1 = data_fold, data_fold2 = data_fold+("MIPS/");

#endif


#if defined(unix) || defined(__unix__)
struct llt
{
	int date, h, m, s;
	llt(size_t diff) { set(diff); }
	void set(size_t diff)
	{
		date = diff / 86400;
		diff = diff % 86400;
		h = diff / 3600;
		diff = diff % 3600;
		m = diff / 60;
		s = diff % 60;
	}
};
#endif

bool should_stop(const SearchState& search_state)
{
	return search_state.candidates_verified >= search_state.candidate_budget;
}

void saveAndShow(float c, int k, std::string& dataset, std::vector<resOutput>& res)
{
	time_t now = time(0);
	tm* ltm = new tm[1];
	localtime_s(ltm, &now);
	std::string query_result = ("results/Running_result.txt");
	std::ofstream os(query_result, std::ios_base::app);
	os.seekp(0, std::ios_base::end); // 

	//time_t now = std::time(0);
	time_t zero_point = 1635153971 - 17 * 3600 - 27 * 60;//Let me set the time at 2021.10.25. 17:27 as the zero point
	size_t diff = (size_t)(now - zero_point);
#if defined(unix) || defined(__unix__)
	llt lt(diff);
#endif

	double date = ((float)(now - zero_point)) / 86400;
	float hour = date - floor(date);
	hour *= 24;
	float minute= hour = date - floor(date);


	std::stringstream ss;

	ss << "*******************************************************************************************************\n"
		<< "The result of FARGO for " << dataset << " is as follow: c="<<c<<", k="<<k
		<<"\n"
		<< "*******************************************************************************************************\n";

	ss << std::setw(12) << "algName"
		<< std::setw(12) << "c"
		<< std::setw(12) << "L"
		<< std::setw(12) << "K"
		<< std::setw(12) << "Time"
		<< std::setw(12) << "Recall"
		<< std::setw(12) << "Ratio"
		<< std::setw(12) << "kRatio"
		<< std::endl
		<< std::endl;
	for (int i = 0; i < res.size(); ++i) {
		ss << std::setw(12) << res[i].algName
			<< std::setw(12) << res[i].c
			<< std::setw(12) << res[i].L
			<< std::setw(12) << res[i].K
			<< std::setw(12) << res[i].time
			<< std::setw(12) << res[i].recall
			<< std::setw(12) << res[i].ratio
			<< std::setw(12) << res[i].kRatio
			<< std::endl;
	}
#if defined(unix) || defined(__unix__)
	ss << "\n******************************************************************************************************\n"
		<< "                                                                                    "
		<< lt.date << '-' << lt.h << ':' << lt.m << ':' << lt.s
		<< "\n******************************************************************************************************\n\n\n";
#else
	ss << "\n******************************************************************************************************\n"
		<< "                                                                                    "
		<< ltm->tm_mon + 1 << '-' << ltm->tm_mday << ' ' << ltm->tm_hour << ':' << ltm->tm_min
		<< "\n*****************************************************************************************************\n\n\n";
#endif
	

	std::cout << ss.str();
	os << ss.str();
	os.close();  delete []ltm;
	//delete[] ltm;
}
