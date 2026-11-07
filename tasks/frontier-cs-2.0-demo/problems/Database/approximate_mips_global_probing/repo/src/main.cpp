#include "query_core.h"
#include <iostream>
int main(){auto c=make_query_core();double p[]={.2,-.4,.9};c->reset(1,3,p);Probe b;while(c->next(b))std::cout<<b.table<<" "<<b.key<<" "<<b.score<<"\n";}
