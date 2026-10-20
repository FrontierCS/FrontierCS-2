Regions Sort: A Parallel Work-Efficient In-Place Radix Sort
======================


Organization
--------

The high level code for MSD radix sort is present in radixSort.h
The code for Regions Sort is present in graphbased.h.
Two implementations for global sorting provided.
The 2-path finding approach is in edgelistgraph.h.
The cycle finding apprach is in cyclegraph.h
radix\_config.h has all the configuration needed for the algorithm.

Compilation
--------

Compilation is done from the directory containing the Makefile.

Compilers

* g++ &gt;= 7.3.1 with support for Cilk Plus

To compile with g++ using Cilk Plus, define the environment variable
CILK.
 
Note: If you
experience any errors, please send an email to [Julian Shun](mailto:jshun@mit.edu), [Endrias Kahssay](mailto:endrias@mit.edu), and [Omar Obeya](mailto:omarobeya@gmail.com).

After the appropriate environment variables are set, to compile,
simply run

```
$ make  #compiles
```

The following commands cleans the directory:
```
$ make clean #removes all executables
```

Running code
-------
An example code is provided in example.cpp.

The sort function takes a pointer to an array, the length of the array, and a function mapping the array to a key.

The array could be an array of integers or pairs of integers.

To set number of Cilk workers to be N use

export CILK\_NWORKERS=N; 

before running the program.

The "-r" flag specifies the number of rounds to run the algorithm. The "-c" flag specfies that a correctness test should be run.

On NUMA machines, adding the command "numactl -i all " when running
the program may improve performance for large number of threads.

Here is an example command for running the code for 3 rounds with correctness checking:

```
numactl -i all ./radixSort -r 3 -c <input_file> 
```

radixSort is the executable that uses the 2-path finding algorithm,
while radixSort\_cycle is the executable that uses the cycle finding
algorithm.

The input file needed by example.cpp is of the following format. For a
sequence of integers, the file should have a header "sequenceInt" on
the first line, followed by one integer per line. For pairs of
integers, the file should have a header "sequenceIntPair", followed by
two integers per line.



Compiler Flags
---------
radix\_config.h has all the configuration needed for the algorithm.

1. CYCLE

Define the macro CYCLE to use the cycle finding global sorting
algorithm, otherwise the code uses the 2-path finding algorithm by default.

2. MAX\_RADIX

Set the MAX\_RADIX macro to the number of bits need to be read at each level of recursion in radix sort.
The code is optimized for 8 bits.



Resources  
-------- 

Omar Obeya, Endrias Kahssay, Edward Fan, and Julian Shun. Theoretically-efficient and practical parallel in-place radix sorting. In ACM Symposium on
Parallelism in Algorithms and Architectures (SPAA), 2019.
