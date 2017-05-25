#include <agency/agency.hpp>
#include <agency/cuda.hpp>
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

  {
    // default construction

    cuda::multidevice_executor exec;

    // create a container to hold the value we'll atomically increment
    std::vector<int,cuda::allocator<int>> result(1,0);

    int n = (1 << 20) + 13;

    // count the number of agents created by bulk_invoke()
    bulk_invoke(cuda::par(n).on(exec), functor{result.data()});

    assert(result[0] == n);
  }

  {
    // from all_devices()

    cuda::multidevice_executor exec = cuda::all_devices();

    // create a container to hold the value we'll atomically increment
    std::vector<int,cuda::allocator<int>> result(1,0);

    int n = (1 << 20) + 13;

    // count the number of agents created by bulk_invoke()
    bulk_invoke(cuda::par(n).on(exec), functor{result.data()});

    assert(result[0] == n);
  }

  {
    // from devices(0)

    cuda::multidevice_executor exec = cuda::devices(0);

    // create a container to hold the value we'll atomically increment
    std::vector<int,cuda::allocator<int>> result(1,0);

    int n = (1 << 20) + 13;

    // count the number of agents created by bulk_invoke()
    bulk_invoke(cuda::par(n).on(exec), functor{result.data()});

    assert(result[0] == n);
  }

  if(cuda::all_devices().size() > 1)
  {
    // from devices(0,1)

    cuda::multidevice_executor exec = cuda::devices(0,1);

    // create a container to hold the value we'll atomically increment
    std::vector<int,cuda::allocator<int>> result(1,0);

    int n = (1 << 20) + 13;

    // count the number of agents created by bulk_invoke()
    bulk_invoke(cuda::par(n).on(exec), functor{result.data()});

    assert(result[0] == n);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

