#include <agency/new_executor_traits.hpp>
#include <cassert>
#include <iostream>

struct my_executor {};

int main()
{
  {
    // void -> void
    my_executor exec;

    auto void_future = agency::when_all();

    int set_me_to_thirteen = 0;

    auto f = agency::new_executor_traits<my_executor>::then_execute(exec, void_future, [&]
    {
      set_me_to_thirteen = 13;
    });

    f.wait();

    assert(set_me_to_thirteen == 13);
  }

  {
    // void -> int
    my_executor exec;

    auto void_future = agency::when_all();

    auto f = agency::new_executor_traits<my_executor>::then_execute(exec, void_future, []
    {
      return 13;
    });

    assert(f.get() == 13);
  }

  {
    // int -> void
    my_executor exec;

    auto int_future = agency::new_executor_traits<my_executor>::make_ready_future<int>(exec, 13);

    int set_me_to_thirteen = 0;

    auto f = agency::new_executor_traits<my_executor>::then_execute(exec, int_future, [&](int& x)
    {
      set_me_to_thirteen = x;
    });

    f.wait();

    assert(set_me_to_thirteen == 13);
  }

  {
    // int -> float
    my_executor exec;

    auto int_future = agency::new_executor_traits<my_executor>::make_ready_future<int>(exec, 13);

    auto f = agency::new_executor_traits<my_executor>::then_execute(exec, int_future, [](int &x)
    {
      return float(x) + 1.f;
    });

    assert(f.get() == 14.f);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

