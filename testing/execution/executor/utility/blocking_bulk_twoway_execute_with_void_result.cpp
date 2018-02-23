#include <agency/agency.hpp>
#include <agency/execution/executor/detail/utility.hpp>
#include <cassert>
#include <iostream>

#include "../test_executors.hpp"

template<class Executor>
void test(Executor exec)
{
  std::atomic<int> counter{0};

  using shape_type = agency::executor_shape_t<Executor>;
  using index_type = agency::executor_index_t<Executor>;

  shape_type shape{10};
  
  agency::detail::blocking_bulk_twoway_execute_with_void_result(exec,
    [&](index_type, int& shared_arg)
    {
      counter += shared_arg;
    },
    shape,
    []{ return 13; } // shared_arg
  );
  
  assert(counter == 13 * 10);
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

