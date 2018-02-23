#include <iostream>
#include <type_traits>
#include <vector>
#include <cassert>

#include <agency/future.hpp>
#include <agency/execution/executor/detail/execution_functions/bulk_then_execute.hpp>
#include <agency/execution/executor/customization_points/make_ready_future.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>

#include "../test_executors.hpp"


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [](index_type idx, int& predecessor, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(std::vector<int>(10, 7 + 13) == result);
}


template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  auto predecessor_future = agency::make_ready_future<void>(exec);

  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute(exec,
    [](index_type idx, std::vector<int>& results, std::vector<int>& shared_arg)
    {
      results[idx] = shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },     // results
    [=]{ return std::vector<int>(shape, 13); }  // shared_arg
  );
  
  auto result = f.get();
  
  assert(std::vector<int>(10, 13) == result);
}


int main()
{
  test_with_non_void_predecessor(bulk_twoway_executor());
  test_with_non_void_predecessor(bulk_then_executor());
  test_with_non_void_predecessor(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor(not_a_bulk_then_executor());
  test_with_non_void_predecessor(complete_bulk_executor());

  test_with_void_predecessor(bulk_twoway_executor());
  test_with_void_predecessor(bulk_then_executor());
  test_with_void_predecessor(not_a_bulk_twoway_executor());
  test_with_void_predecessor(not_a_bulk_then_executor());
  test_with_void_predecessor(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

