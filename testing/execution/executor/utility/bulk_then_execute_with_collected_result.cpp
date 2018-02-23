#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <iostream>

#include "../test_executors.hpp"

template<class Executor>
void test_with_void_predecessor(Executor exec)
{
  using index_type = agency::executor_index_t<Executor>;

  auto predecessor_future = agency::make_ready_future<void>(exec);

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();
  
  assert(std::vector<int>(shape, 13) == result);
}


template<class Executor>
void test_with_non_void_predecessor(Executor exec)
{
  using index_type = agency::executor_index_t<Executor>;

  auto predecessor_future = agency::make_ready_future<int>(exec, 7);

  size_t shape = 10;
  
  auto f = agency::detail::bulk_then_execute_with_collected_result(exec,
    [](index_type idx, int& predecessor, std::vector<int>& shared_arg)
    {
      return predecessor + shared_arg[idx];
    },
    shape,
    predecessor_future,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );

  auto result = f.get();
  
  assert(std::vector<int>(shape, 7 + 13) == result);
}

int main()
{
  test_with_void_predecessor(bulk_twoway_executor());
  test_with_void_predecessor(bulk_then_executor());
  test_with_void_predecessor(not_a_bulk_twoway_executor());
  test_with_void_predecessor(not_a_bulk_then_executor());
  test_with_void_predecessor(complete_bulk_executor());

  test_with_non_void_predecessor(bulk_twoway_executor());
  test_with_non_void_predecessor(bulk_then_executor());
  test_with_non_void_predecessor(not_a_bulk_twoway_executor());
  test_with_non_void_predecessor(not_a_bulk_then_executor());
  test_with_non_void_predecessor(complete_bulk_executor());

  std::cout << "OK" << std::endl;

  return 0;
}

