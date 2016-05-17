#include <agency/experimental/array.hpp>
#include <agency/experimental/view.hpp>
#include <agency/experimental/strided_view.hpp>
#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <vector>
#include <cassert>
#include <iostream>


template<class Range>
__host__ __device__
int sequential_sum(Range&& rng)
{
  int result = 0;

  for(auto i = rng.begin(); i != rng.end(); ++i)
  {
    result += *i;
  }

  return result;
}


template<class ConcurrentAgent, class Range1, class Range2>
__host__ __device__
int concurrent_sum(ConcurrentAgent& self, Range1&& rng, Range2&& scratch)
{
  using namespace agency::experimental;

  auto i = self.index();

  // each agent strides through the range sequentially and sums up a partial sum into scratch
  scratch[i] = sequential_sum(strided(drop(rng, i), self.group_size()));

  self.wait();

  auto n = scratch.size();
  
  while(n > 1)
  {
    if(i < n/2)
    {
      scratch[i] += scratch[n - i - 1];
    }
  
    // wait for every agent in the group to reach this point
    self.wait();
  
    // cut the number of active agents in half
    n -= n/2;
  }
  
  return scratch[0];
}


template<class View>
agency::cuda::future<int> async_reduce(View data)
{
  using namespace agency;
  using namespace agency::experimental;

  constexpr size_t tile_size = 256;
  using tile = array<int,tile_size>;

  // XXX the size of each partition needn't match the inner group size
  // XXX we should use the shape of the executor for partitioning instead of the data size
  size_t num_tiles = (data.size() + (tile_size - 1)) / tile_size;

  auto policy = cuda::par(num_tiles, cuda::con(tile_size));

  auto partial_sums_fut = bulk_async(policy,
    [=] __host__ __device__ (parallel_group<cuda::concurrent_agent>& self, tile& scratch) -> scope_result<1,int>
    {
      // find this group's partition
      auto partition = data.subspan(tile_size * self.outer().index(), tile_size);

      // sum the partition
      auto partial_sum = concurrent_sum(self.inner(), partition, scratch);

      // the first agent returns the partial_sum
      if(self.inner().index() == 0)
      {
        return partial_sum;
      }

      // other agents return no result
      return no_result<int>();
    },
    share_at_scope<1,tile>()
  );

  return bulk_then(policy.inner(),
    [] __host__ __device__ (cuda::concurrent_agent& self, span<int> partial_sums, tile& scratch) -> single_result<int>
    {
      auto sum = concurrent_sum(self, partial_sums, scratch);

      // the first agent returns the sum
      if(self.index() == 0)
      {
        return sum;
      }

      // other agents return no result
      return no_result<int>();
    },
    partial_sums_fut,
    share<tile>()
  );
}


int main()
{
  size_t n = 1 << 20;

  std::vector<int, agency::cuda::allocator<int>> data(n, 1);

  auto result = async_reduce(agency::experimental::all(data)).get();

  assert(result == n);

  std::cout << "OK" << std::endl;

  return 0;
}

