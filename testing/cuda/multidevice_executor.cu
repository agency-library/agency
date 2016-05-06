#include <agency/bulk_invoke.hpp>
#include <agency/cuda/execution_policy.hpp>
#include <agency/cuda/executor/multidevice_executor.hpp>
#include <vector>

struct functor
{
  int *ptr;

  template<class Agent>
  __device__
  void operator()(Agent& self)
  {
    atomicAdd(ptr, 1);
  }
};

int main()
{
  using namespace agency;

  cuda::multidevice_executor exec;

  // create a container to hold the value we'll atomically increment
  using container = cuda::multidevice_executor::container<int>;
  container result(1,0);

  size_t n = (1 << 20) + 13;

  // count the number of agents created by bulk_invoke()
  bulk_invoke(cuda::par(n).on(exec), functor{result.data()});

  assert(result[0] == n);

  std::cout << "OK" << std::endl;

  return 0;
}

