#include <agency/agency.hpp>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

// adapted from https://github.com/chriskohlhoff/executors/blob/master/src/examples/executor/fork_join.cpp

template<class Iterator>
void parallel_sort_recursive(Iterator first, Iterator last)
{
  auto n = last - first;
  if(n > 32768)
  {
    auto num_agents = 2;
    auto partition_size = n / num_agents;

    bulk_invoke(agency::par(2), [=](agency::parallel_agent& self)
    {
      auto begin = std::min(last, first + self.index() * partition_size);
      auto end   = std::min(last, begin + partition_size);

      parallel_sort_recursive(begin, end);
    });

    // XXX only works for num_agents == 2
    //     generalization would be to bulk_invoke half as many agents
    //     and do that many inplace_merges
    std::inplace_merge(first, first + partition_size, last);
  }
  else
  {
    std::sort(first, last);
  }
}

template<class T1, class T2>
auto ceil_div(T1 n, T2 d)
  -> decltype(
       (n + (d - T2(1))) / d
     )
{
  return (n + (d - T2(1))) / d;
}

template<class RandomAccessIterator>
void parallel_sort_iterative(RandomAccessIterator first, RandomAccessIterator last)
{
  auto n = last - first;
  auto partition_size = 32768;
  auto num_partitions = ceil_div(n, partition_size);

  bulk_invoke(agency::par(num_partitions), [=](agency::parallel_agent& self)
  {
    auto begin = std::min(last, first + self.index() * partition_size);
    auto end   = std::min(last, begin + partition_size);

    std::sort(begin, end);
  });

  for(; partition_size < n; partition_size *= 2)
  {
    auto num_partitions = ceil_div(n, partition_size);

    bulk_invoke(agency::par(num_partitions / 2), [=](agency::parallel_agent& self)
    {
      auto begin = std::min(last, first + 2 * self.index() * partition_size);
      auto mid   = std::min(last, begin + partition_size);
      auto end   = std::min(last, mid + partition_size);

      std::inplace_merge(begin, mid, end);
    });
  }
}

int main()
{
  auto n = 1 << 20;
  std::vector<int> data(n);

  std::default_random_engine rng;

  std::generate(data.begin(), data.end(), rng);

  // ensure unsorted data
  data[0] = 1;
  data[1] = 0;

  assert(!std::is_sorted(data.begin(), data.end()));

  parallel_sort_recursive(data.begin(), data.end());

  assert(std::is_sorted(data.begin(), data.end()));

  std::generate(data.begin(), data.end(), rng);

  // ensure unsorted data
  data[0] = 1;
  data[1] = 0;

  assert(!std::is_sorted(data.begin(), data.end()));

  parallel_sort_iterative(data.begin(), data.end());

  assert(std::is_sorted(data.begin(), data.end()));

  std::cout << "OK" << std::endl;

  return 0;
}

