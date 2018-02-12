#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/future.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <agency/execution/executor/detail/utility/blocking_bulk_twoway_execute.hpp>

#include "../test_executors.hpp"


template<class Executor>
void test(Executor exec)
{
  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape = 10;
  
  auto result = agency::detail::blocking_bulk_twoway_execute(exec,
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
  test(bulk_twoway_executor());
  test(bulk_then_executor());

  test(not_a_bulk_twoway_executor());
  test(not_a_bulk_then_executor());

  test(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}


