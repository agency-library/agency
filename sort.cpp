#include <execution_policy>
#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>

template<class Iterator>
void parallel_sort(Iterator first, Iterator last)
{
  auto n = last - first;
  if(n > 32768)
  {
    auto num_agents = 2;
    auto partition_size = n / num_agents;

    bulk_invoke(std::par(2), [=](std::parallel_agent& self)
    {
      auto begin = std::min(last, first + self.index() * partition_size);
      auto end   = std::min(last, begin + partition_size);

      parallel_sort(begin, end);
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

  parallel_sort(data.begin(), data.end());

  assert(std::is_sorted(data.begin(), data.end()));

  std::cout << "OK" << std::endl;

  return 0;
}

