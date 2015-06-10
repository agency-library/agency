#include <agency/new_executor_traits.hpp>
#include <cassert>
#include <iostream>

struct my_executor {};

int main()
{
  {
    // returning void
    my_executor exec;

    int set_me_to_thirteen = 0;

    auto f = agency::new_executor_traits<my_executor>::async_execute(exec, [&]
    {
      set_me_to_thirteen = 13;
    });

    f.wait();

    assert(set_me_to_thirteen == 13);
  }

  {
    // returning int
    my_executor exec;

    auto f = agency::new_executor_traits<my_executor>::async_execute(exec, []
    {
      return 13;
    });

    assert(f.get() == 13);
  }

  std::cout << "OK" << std::endl;

  return 0;
}


