#include <agency/agency.hpp>


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
      []()
    {
      return 7;
    });

    assert(result == 7);
  }

  {
    // invoke with one parameter

    int val = 13;

    auto result = agency::invoke(executor,
      [](int val)
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
      [](int val1, int val2)
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

  std::cout << "OK" << std::endl;

  return 0;
}

