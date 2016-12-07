#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <cstdio>


struct kernel
{
  __device__
  void operator()(agency::parallel_group<agency::concurrent_agent>& self)
  {
    printf("hello from {%d, %d}\n", (int)self.outer().index(), (int)self.inner().index());
  }

  __device__
  void operator()(agency::parallel_group_2d<agency::concurrent_agent_2d>& self)
  {
    auto outer_idx = self.outer().index();
    auto inner_idx = self.inner().index();

    printf("hello from {{%d, %d}, {%d, %d}}\n", (int)outer_idx[0], (int)outer_idx[1], (int)inner_idx[0], (int)inner_idx[1]);
  }
};


auto grid(size_t num_blocks, size_t num_threads)
  -> decltype(
       agency::cuda::par(num_blocks, agency::cuda::con(num_threads))
     )
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}


auto grid(agency::size2 num_blocks, agency::size2 num_threads)
  -> decltype(
       agency::cuda::par2d(num_blocks, agency::cuda::con2d(num_threads))
     )
{
  return agency::cuda::par2d(num_blocks, agency::cuda::con2d(num_threads));
}


int main()
{
  agency::bulk_invoke(grid(2,32), kernel());

  agency::bulk_invoke(grid({1,2}, {1,32}), kernel());

  return 0;
}

