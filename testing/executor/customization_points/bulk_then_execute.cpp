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

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::bulk_then_execute(exec,
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

  using shape_type = agency::new_executor_shape_t<Executor>;
  using index_type = agency::new_executor_index_t<Executor>;

  size_t shape = 10;
  
  auto f = agency::bulk_then_execute(exec,
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
  test_with_non_void_predecessor(bulk_synchronous_executor());
  test_with_non_void_predecessor(bulk_asynchronous_executor());
  test_with_non_void_predecessor(bulk_continuation_executor());
  test_with_non_void_predecessor(not_a_bulk_synchronous_executor());
  test_with_non_void_predecessor(not_a_bulk_asynchronous_executor());
  test_with_non_void_predecessor(not_a_bulk_continuation_executor());
  test_with_non_void_predecessor(complete_bulk_executor());

  test_with_void_predecessor(bulk_synchronous_executor());
  test_with_void_predecessor(bulk_asynchronous_executor());
  test_with_void_predecessor(bulk_continuation_executor());
  test_with_void_predecessor(not_a_bulk_synchronous_executor());
  test_with_void_predecessor(not_a_bulk_asynchronous_executor());
  test_with_void_predecessor(not_a_bulk_continuation_executor());
  test_with_void_predecessor(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

