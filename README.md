What is Agency?
===============

Agency is an experimental C++ template library for parallel programming. Unlike
higher-level parallel algorithms libraries like [Thrust](thrust.github.io),
Agency provides **lower-level** primitives for **creating execution**. Agency
interoperates with standard components like **execution policies** and
**executors** to enable the creation of **portable** parallel algorithms.

# Examples

Agency is best-explained through examples. The following program implements a parallel sum.

~~~~{.cpp}
#include <agency/agency.hpp>
#include <agency/experimental.hpp>
#include <vector>
#include <numeric>
#include <iostream>
#include <cassert>

int parallel_sum(int* data, int n)
{
  // create a view of the input
  agency::experimental::span<int> input(data, n);

  // divide the input into 8 chunks
  int num_agents = 8;
  auto chunks = agency::experimental::chunk_evenly(input, num_agents);

  // create 8 agents to sum each chunk in parallel
  auto partial_sums = agency::bulk_invoke(agency::par(num_agents), [=](agency::parallel_agent& self)
  {
    // get this parallel agent's chunk
    auto this_chunk = chunks[self.index()];

    // return the sum of this chunk
    return std::accumulate(this_chunk.begin(), this_chunk.end(), 0);
  });

  // return the sum of partial sums
  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0);
}

int main()
{
  // create a large vector filled with 1s
  std::vector<int> vec(32 << 20, 1);

  int sum = parallel_sum(vec.data(), vec.size());

  std::cout << "sum is " << sum << std::endl;

  assert(sum == vec.size());

  return 0;
}
~~~~

# Discover the Library

* Refer to Agency's [Quick Start Guide](http://github.com/jaredhoberock/agency/wiki/Quick-Start-Guide) for further information and examples.
* See Agency in action in the [collection of example programs](http://github.com/jaredhoberock/agency/tree/master/examples).
* Browse Agency's online API documentation.

