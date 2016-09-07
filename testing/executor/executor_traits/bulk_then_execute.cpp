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

  auto fut = agency::detail::make_ready_future<int>(7);
  
  int val = 13;

  using shape_type = executor_shape_t<Executor>;
  using index_type = executor_index_t<Executor>;
  
  auto f = bulk_then_execute(exec,
    [](index_type idx, int& past_arg, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = past_arg + shared_arg[idx];
    },
    10,
    fut,
    [](shape_type shape){ return std::vector<int>(shape); },     // results
    [](shape_type shape){ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(std::vector<int>(10, 7 + 13) == result);
}


int main()
{
  test(bulk_synchronous_executor());
  test(bulk_asynchronous_executor());
  test(bulk_continuation_executor());

  test(not_a_bulk_synchronous_executor());
  test(not_a_bulk_asynchronous_executor());
  test(not_a_bulk_continuation_executor());

  test(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

