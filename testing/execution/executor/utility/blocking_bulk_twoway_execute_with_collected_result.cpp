#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <iostream>

#include "../test_executors.hpp"

template<class Executor>
void test(Executor exec)
{
  using index_type = agency::executor_index_t<Executor>;

  size_t shape = 10;
  
  auto result = agency::detail::blocking_bulk_twoway_execute_with_collected_result(exec,
    [](index_type idx, std::vector<int>& shared_arg)
    {
      return shared_arg[idx];
    },
    shape,
    [=]{ return std::vector<int>(shape); },    // results
    [=]{ return std::vector<int>(shape, 13); } // shared_arg
  );
  
  assert(std::vector<int>(shape, 13) == result);
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

