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

int main()
{
  using namespace agency::cuda;

  bulk_invoke(par(2,con(32)), kernel());

  return 0;
}

