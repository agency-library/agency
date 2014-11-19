#include <iostream>
#include <agency/execution_policy.hpp>
#include "cuda/execution_policy.hpp"

struct functor
{
  __host__ __device__
  void operator()(cuda::concurrent_agent& self)
  {
    printf("agent %d arriving at barrier\n", (int)self.index());

    self.wait();

    printf("departing barrier\n");
  }
};

int main()
{
  cuda::block_executor gpu;
  
  agency::bulk_invoke(agency::con(10).on(gpu), functor());

  return 0;
}

