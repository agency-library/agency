#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <cassert>


void test()
{
  {
    // invoke with no parameters

    auto result = agency::invoke([]()
    {
      return 7;
    });

    assert(result == 7);
  }

  {
    // invoke with one parameter

    int val = 13;

    auto result = agency::invoke([](int val)
    {
      return val;
    },
    val);

    assert(result == 13);
  }

  {
    // invoke with two parameters

    int val1 = 13, val2 = 7;

    auto result = agency::invoke([](int val1, int val2)
    {
      return val1 + val2;
    },
    val1, val2);

    assert(result == val1 + val2);
  }
}

template<class Executor>
void test(Executor executor)
{
  {
    // invoke with no parameters

    auto result = agency::invoke(executor,
      [] __host__ __device__ ()
    {
      return 7;
    });

    assert(result == 7);
  }

  {
    // invoke with one parameter

    int val = 13;

    auto result = agency::invoke(executor,
      [] __host__ __device__ (int val)
    {
      return val;
    },
    val);

    assert(result == 13);
  }

  {
    // invoke with two parameters

    int val1 = 13, val2 = 7;

    auto result = agency::invoke(executor,
      [] __host__ __device__ (int val1, int val2)
    {
      return val1 + val2;
    },
    val1, val2);

    assert(result == val1 + val2);
  }
}

int main()
{
  using namespace agency;

  test();

  test(sequenced_executor());
  test(concurrent_executor());
  test(parallel_executor());

  test(cuda::concurrent_executor());
  test(cuda::parallel_executor());
  test(cuda::grid_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

