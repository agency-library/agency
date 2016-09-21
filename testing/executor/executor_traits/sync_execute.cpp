#include <iostream>
#include <type_traits>
#include <vector>

#include <agency/future.hpp>
#include <agency/execution/executor/detail/new_executor_traits.hpp>

#include "test_executors.hpp"


template<class Executor>
void test(Executor exec)
{
  using namespace agency::detail::new_executor_traits_detail;
  
  auto result = sync_execute(exec, []{ return 7; });
  
  assert(7 == result);
}


int main()
{
  test(continuation_executor());
  test(asynchronous_executor());
  test(synchronous_executor());

  test(bulk_continuation_executor());
  test(bulk_asynchronous_executor());
  test(bulk_synchronous_executor());

  // XXX call test() with all the other types of executors

  std::cout << "OK" << std::endl;

  return 0;
}
