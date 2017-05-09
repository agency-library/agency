#include <agency/agency.hpp>
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


template<class Executor>
void test(Executor executor)
{
  {
    // invoke with no parameters

    std::atomic<int> counter{0};

    agency::invoke(executor, [&]()
    {
      ++counter;
    });

    assert(counter == 1);
  }

  {
    // invoke with one parameter

    int val = 13;

    std::atomic<int> counter{0};

    agency::invoke(executor,
      [&](int val)
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

    std::atomic<int> counter{0};

    agency::invoke(executor,
      [&](int val1, int val2)
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

  std::cout << "OK" << std::endl;

  return 0;
}

