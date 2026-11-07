#include "edlib.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
// Judge-owned API probe: argv[1] names a case file, one case per line
// "<query.fa> <target.fa> <mode> <task> <k> <ambiguous>"; one edlibAlign call and one JSON
// line per case, in order.
static std::string read(const std::string& p) { std::ifstream f(p); std::string line,s; while(std::getline(f,line)) if(line.empty()||line[0]!='>') s+=line; return s; }
static void align_case(const std::string& qp,const std::string& tp,int mode,int task,int k,int amb) {
 std::string q=read(qp),t=read(tp);
 EdlibEqualityPair eq[]={{'N','A'},{'N','C'},{'N','G'},{'N','T'}};
 auto c=edlibNewAlignConfig(k,static_cast<EdlibAlignMode>(mode),static_cast<EdlibAlignTask>(task),amb?eq:NULL,amb?4:0);
 auto r=edlibAlign(q.data(),q.size(),t.data(),t.size(),c);
 std::cout<<"{\"status\":"<<r.status<<",\"distance\":"<<r.editDistance<<",\"ends\":[";
 for(int i=0;i<r.numLocations;i++){if(i)std::cout<<',';std::cout<<r.endLocations[i];}
 std::cout<<"],\"starts\":[";
 if(r.startLocations)for(int i=0;i<r.numLocations;i++){if(i)std::cout<<',';std::cout<<r.startLocations[i];}
 std::cout<<"],\"ops\":\"";
 if(r.alignment)for(int i=0;i<r.alignmentLength;i++)std::cout<<int(r.alignment[i]);
 std::cout<<"\",\"length\":"<<r.alignmentLength<<",\"cigar\":\"";
 char* cigar=r.alignment?edlibAlignmentToCigar(r.alignment,r.alignmentLength,EDLIB_CIGAR_EXTENDED):NULL;
 if(cigar)std::cout<<cigar;
 std::cout<<"\"}\n"; free(cigar); edlibFreeAlignResult(r);
}
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::ifstream cases(argv[1]);
 std::string qp,tp; int mode,task,k,amb;
 while(cases>>qp>>tp>>mode>>task>>k>>amb) align_case(qp,tp,mode,task,k,amb);
 if(!cases.eof())return 2;
 std::cout.flush(); return std::cout?0:1;
}
