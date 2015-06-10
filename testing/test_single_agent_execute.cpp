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

    agency::new_executor_traits<my_executor>::execute(exec, [&]
    {
      set_me_to_thirteen = 13;
    });

    assert(set_me_to_thirteen == 13);
  }

  {
    // returning int
    my_executor exec;

    auto result = agency::new_executor_traits<my_executor>::execute(exec, []
    {
      return 13;
    });

    assert(result == 13);
  }

  std::cout << "OK" << std::endl;

  return 0;
}


