#include <agency/agency.hpp>
#include <atomic>


void test()
{
  {
    // async with no parameters

    std::atomic<int> counter{0};

    auto f = agency::async([&]()
    {
      ++counter;
    });

    f.wait();

    assert(counter == 1);
  }

  {
    // async with one parameter

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::async([&](int val)
    {
      counter += val;
    },
    val);

    f.wait();

    assert(counter == 13);
  }

  {
    // async with two parameters

    int val1 = 13, val2 = 7;

    std::atomic<int> counter{0};

    auto f = agency::async([&](int val1, int val2)
    {
      counter += val1 + val2;
    },
    val1, val2);

    f.wait();

    assert(counter == val1 + val2);
  }
}


template<class Executor>
void test(Executor executor)
{
  {
    // async with no parameters

    std::atomic<int> counter{0};

    auto f = agency::async(executor, [&]()
    {
      ++counter;
    });

    f.wait();

    assert(counter == 1);
  }

  {
    // async with one parameter

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::async(executor,
      [&](int val)
      {
        counter += val;
      },
      val
    );

    f.wait();

    assert(counter == 13);
  }

  {
    // async with two parameters

    int val1 = 13, val2 = 7;

    std::atomic<int> counter{0};

    auto f = agency::async(executor,
      [&](int val1, int val2)
      {
        counter += val1 + val2;
      },
      val1,
      val2
    );

    f.wait();

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

