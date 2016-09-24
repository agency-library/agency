#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/execution/executor/customization_points.hpp>

#include "../test_executors.hpp"


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 7);
  
  auto f = agency::then_execute(exec, [](int& predecessor)
  {
    return predecessor + 13;
  },
  predecessor_future);
  
  auto result = f.get();
  
  assert(7 + 13 == result);
}


template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<void>(exec);

  auto f = agency::then_execute(exec, []{ return 13; }, predecessor_future);
  
  auto result = f.get();
  
  assert(13 == result);
}


int main()
{
  test_with_non_void_predecessor(continuation_executor());
  test_with_non_void_predecessor(bulk_continuation_executor());
  // XXX call test_with_non_void_predecessor() with all the other types of executors

  test_with_void_predecessor(continuation_executor());
  test_with_void_predecessor(bulk_continuation_executor());
  // XXX call test_with_void_predecessor() with all the other types of executors

  std::cout << "OK" << std::endl;

  return 0;
}

