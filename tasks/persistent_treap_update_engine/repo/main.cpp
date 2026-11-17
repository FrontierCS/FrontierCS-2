
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "utils/log.h"
#include "utils/timer.h"

#include "db/include/interface.hpp"
#include "db/batch.hpp"
#include "db/contreap.hpp"
#include "db/sequential.hpp"
#include "lib/pam/interface.hpp"

void ExitWithHint(const char* command) {
  std::cerr << "Usage: " << command << " <method> <workload> [options]" << std::endl;
  std::cerr << "Methods: contreap, sequential, pam" << std::endl;
  std::cerr << "Options:" << std::endl;
  std::cerr << "  -threads n: number of server side threads (default: number of CPU cores)" << std::endl;
  std::cerr << "  -clients n: number of client side threads (default: 1)" << std::endl;
  std::cerr << "  -batchsize b: specify the batch size (default: 1000)" << std::endl;
  exit(0);
}

std::string dbname;
std::string filename;
size_t num_threads = std::thread::hardware_concurrency();
size_t num_clients = 1;
size_t batch_size = 1000;

void ParseCommandLine(int argc, const char *argv[]) {
  if (argc < 3) { ExitWithHint(argv[0]); }
  dbname = argv[1];
  filename = argv[2];
  size_t argindex = 3;
  while (argindex < argc && strncmp(argv[argindex], "-", 1) == 0) {
    if (strcmp(argv[argindex], "-threads") == 0) {
      argindex++;
      if (argindex >= argc) { ExitWithHint(argv[0]); }
      num_threads = std::stoi(argv[argindex]);
      argindex++; }
    else if (strcmp(argv[argindex], "-clients") == 0) {
      argindex++;
      if (argindex >= argc) { ExitWithHint(argv[0]); }
      num_clients = std::stoi(argv[argindex]);
      argindex++; }
    else if (strcmp(argv[argindex], "-batchsize") == 0) {
      argindex++;
      if (argindex >= argc) { ExitWithHint(argv[0]); }
      batch_size = std::stoi(argv[argindex]);
      if (batch_size == 0) { batch_size = 1; }
      argindex++; }
    else {
      log_fatal("Unknown option '%s'", argv[argindex]);
      ExitWithHint(argv[0]); } }
  if (argindex != argc) { ExitWithHint(argv[0]); }
}

struct tx_context {
  uint8_t type;
  uint64_t arg0;
  uint64_t arg1;
};

int main(int argc, const char *argv[]) {
  ParseCommandLine(argc, argv);

  DB::Interface* db;
  if (dbname == "sequential") { db = new DB::Sequential(num_threads, num_clients); }
  else if (dbname == "contreap") { db = new DB::Contreap(num_threads, num_clients, batch_size); }
  else if (dbname == "pam") { db = new DB::Batch<pam::interface>(num_threads, num_clients, batch_size); }
  else {
    log_fatal("Unknown method '%s'", dbname.c_str());
    ExitWithHint(argv[0]); }

  std::ifstream fs(filename, std::ios::binary);
  size_t n, m;
  fs.read((char*)&n, sizeof(n));
  fs.read((char*)&m, sizeof(m));
  uint64_t* elems = new uint64_t[n];
  fs.read((char*)elems, n * sizeof(uint64_t));
  std::cout << "# Loaded records:\t" << n << std::endl;
  tx_context* txs = new tx_context[m];
  fs.read((char*)txs, m * sizeof(tx_context));
  std::cout << "# Loaded transactions:\t" << m << std::endl;

  db->Init(n, m, elems);

  Timer tmr;
  tmr.Start();

  for (size_t i = 0; i < m; i++) {
    switch (txs[i].type) {
      case 0 /* QUERY */: db->Query(txs[i].arg0, txs[i].arg1); break;
      case 1 /* INSERT */: db->Insert(txs[i].arg0); break;
      case 2 /* DELETE */: db->Delete(txs[i].arg0); break;
      default: log_fatal("Unknown Transaction Type %u", (unsigned)txs[i].type); } }
  db->Close();

  double duration = tmr.End();

  std::cout << "# Time Used (S)" << std::endl;
  std::cout << dbname << '\t' << filename << '\t' << num_threads << '\t';
  std::cout << duration << std::endl;

  std::cout << "# Transaction throughput (KTPS)" << std::endl;
  std::cout << dbname << '\t' << filename << '\t' << num_threads << '\t';
  std::cout << m / duration / 1000 << std::endl;

  return 0;
}
