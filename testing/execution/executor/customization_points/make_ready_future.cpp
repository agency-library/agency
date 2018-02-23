#include <iostream>
#include <cassert>
#include <agency/execution/executor/customization_points/make_ready_future.hpp>

#include "../test_executors.hpp"

template<class Executor>
void test(Executor exec)
{
  // make a void future
  {
    auto f = agency::make_ready_future<void>(exec);
    assert(f.valid());
    f.wait();
  }

  // make an int future
  {
    auto f = agency::make_ready_future<int>(exec, 13);
    assert(f.valid());
    assert(f.get() == 13);
  }
}

int main()
{
  test(bulk_twoway_executor());
  test(bulk_then_executor());

  test(not_a_bulk_twoway_executor());
  test(not_a_bulk_then_executor());

  test(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

