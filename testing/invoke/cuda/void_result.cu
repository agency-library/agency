#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <atomic>


void test()
{
  {
    // invoke with no parameters

    std::atomic<int> counter{0};

    agency::invoke([&]()
    {
      ++counter;
    });

    assert(counter == 1);
  }

  {
    // invoke with one parameter

    int val = 13;

    std::atomic<int> counter{0};

    agency::invoke([&](int val)
    {
      counter += val;
    },
    val);

    assert(counter == 13);
  }

  {
    // invoke with two parameters

    int val1 = 13, val2 = 7;

    std::atomic<int> counter{0};

    agency::invoke([&](int val1, int val2)
    {
      counter += val1 + val2;
    },
    val1, val2);

    assert(counter == val1 + val2);
  }
}


__managed__ int counter;


template<class Executor>
void test(Executor executor)
{
  {
    // invoke with no parameters

    counter = 0;

    agency::invoke(executor, [] __host__ __device__ ()
    {
      ++counter;
    });

    assert(counter == 1);
  }

  {
    // invoke with one parameter

    int val = 13;

    counter = 0;

    agency::invoke(executor,
      [] __host__ __device__ (int val)
      {
        counter += val;
      },
      val
    );

    assert(counter == 13);
  }

  {
    // invoke with two parameters

    int val1 = 13, val2 = 7;

    counter = 0;

    agency::invoke(executor,
      [] __host__ __device__ (int val1, int val2)
      {
        counter += val1 + val2;
      },
      val1,
      val2
    );

    assert(counter == val1 + val2);
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

