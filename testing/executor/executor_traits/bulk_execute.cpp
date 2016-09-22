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

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  shape_type shape = 10;
  
  auto result = bulk_execute(exec,
    [](index_type idx, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = 7 + shared_arg[idx];
    },
    shape,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
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


