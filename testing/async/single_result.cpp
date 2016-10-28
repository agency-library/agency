#include <agency/agency.hpp>


void test()
{
  {
    // async with no parameters

    auto f = agency::async([]()
    {
      return 7;
    });

    auto result = f.get();

    assert(result == 7);
  }

  {
    // async with one parameter

    int val = 13;

    auto f = agency::async([](int val)
    {
      return val;
    },
    val);

    auto result = f.get();

    assert(result == 13);
  }

  {
    // async with two parameters

    int val1 = 13, val2 = 7;

    auto f = agency::async([](int val1, int val2)
    {
      return val1 + val2;
    },
    val1, val2);

    auto result = f.get();

    assert(result == val1 + val2);
  }
}

template<class Executor>
void test(Executor executor)
{
  {
    // async with no parameters

    auto f = agency::async(executor,
      []()
    {
      return 7;
    });

    auto result = f.get();

    assert(result == 7);
  }

  {
    // async with one parameter

    int val = 13;

    auto f = agency::async(executor,
      [](int val)
    {
      return val;
    },
    val);

    auto result = f.get();

    assert(result == 13);
  }

  {
    // async with two parameters

    int val1 = 13, val2 = 7;

    auto f = agency::async(executor,
      [](int val1, int val2)
    {
      return val1 + val2;
    },
    val1, val2);

    auto result = f.get();

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

  std::cout << "OK" << std::endl;

  return 0;
}

