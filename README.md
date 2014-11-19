Agency
===============

Agency is an experiment exploring how to marry bulk synchronous parallel programming with the components described in the Technical Specification for C++ Extensions for Parallelism. The programming model Agency embodies is intended to be suited to all parallel architectures and is particularly exploitable by wide architectures exposing fine-grained parallelism.

The `bulk_invoke` function creates groups of execution agents which all invoke a lambda en masse:

    template<class Iterator, class T, class BinaryFunction>
    T reduce(Iterator first, Iterator last, T init, BinaryFunction binary_op)
    {
      using namespace agency;
      auto n = std::distance(first, last);

      // chop up data into partitions
      auto partition_size = 10000;
      auto num_partitions = (n + partition_size - 1) / partition_size;
      std::vector<T> partial_sums(num_partitions);

      bulk_invoke(par(num_partitions), [=,&partial_sums](parallel_agent& g)
      {
        auto i = g.index();

        auto partition_begin = first + partition_size * i;
        auto partition_end   = std::min(last, partition_begin + partition_size);

        partial_sums[i] = reduce(seq, partition_begin + 1, partition_end, *partition_begin, binary_op);
      });

      return reduce(seq, partial_sums.begin(), partial_sums.end(), init, binary_op);
    }

# Design Goals

The design of the library is intended to achieve the following goals:

  * Deliver efficiency by exploiting structured concurrency and sharing

  * Build upon the execution policy approach introduced by the Parallelism TS

  * Interface to the underlying platform via executors

  * Provide a mechanism for controlling the placement of work to be created

# Building the Example Programs

Programs with filenames ending in the `.cpp` extension are compilable with a C++11 compiler, e.g.:

    $ g++ -std=c++11 -I. -pthread example.cpp

or

    $ clang -std=c++11 -I. -pthread -lstdc++ example.cpp
    
Programs with filenames ending in the `.cu` extension are compilable with the NVIDIA compiler, e.g.:

    $ nvcc -std=c++11 -I. example.cu
