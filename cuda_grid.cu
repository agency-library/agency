#include <agency/cuda/execution_policy.hpp>
#include <agency/execution_policy.hpp>
#include <iostream>
#include <typeinfo>

struct kernel
{
  __device__
  void operator()(agency::parallel_group<agency::cuda::concurrent_agent>& self)
  {
    printf("hello from {%d, %d}\n", (int)self.outer().index(), (int)self.inner().index());
  }
};


auto grid(size_t num_blocks, size_t num_threads)
  -> decltype(
       agency::cuda::par(num_blocks, agency::cuda::con(num_threads))
     )
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}


int main()
{
  bulk_invoke(grid(2,32), kernel());

  return 0;
}

