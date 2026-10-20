#include "edlib.h"
#include <fstream>
#include <iostream>
#include <string>
#include <cstdlib>
static std::string read(const char* p) { std::ifstream f(p); std::string line,s; while(std::getline(f,line)) if(line.empty()||line[0]!='>') s+=line; return s; }
int main(int argc,char**argv) {
 if(argc!=7)return 2;
 std::string q=read(argv[1]),t=read(argv[2]);
 EdlibEqualityPair eq[]={{'N','A'},{'N','C'},{'N','G'},{'N','T'}};
 auto c=edlibNewAlignConfig(std::atoi(argv[5]),static_cast<EdlibAlignMode>(std::atoi(argv[3])),static_cast<EdlibAlignTask>(std::atoi(argv[4])),std::atoi(argv[6])?eq:NULL,std::atoi(argv[6])?4:0);
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
 std::cout<<"\"}\n"; free(cigar); edlibFreeAlignResult(r); return 0;
}
