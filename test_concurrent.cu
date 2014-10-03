#include <iostream>
#include <execution_policy>
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
  
  std::bulk_invoke(std::con(10).on(gpu), functor());

  return 0;
}

