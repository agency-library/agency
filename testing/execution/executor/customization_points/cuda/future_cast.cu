#include <iostream>
#include <cassert>
#include <agency/execution/executor/customization_points/make_ready_future.hpp>
#include <agency/execution/executor/customization_points/future_cast.hpp>

#include "../../test_executors.hpp"

struct then_executor_with_future_cast : then_executor
{
  then_executor_with_future_cast()
    : function_called_(false)
  {}

  ~then_executor_with_future_cast()
  {
    assert(function_called_);
  }

  template<class T, class Future>
  std::future<T> future_cast(Future& fut) const
  {
    function_called_ = true;
    return agency::future_traits<Future>::template cast<T>(fut);
  }

  mutable bool function_called_;
};


struct bulk_then_executor_with_future_cast : bulk_then_executor
{
  bulk_then_executor_with_future_cast()
    : function_called_(false)
  {}

  ~bulk_then_executor_with_future_cast()
  {
    assert(function_called_);
  }

  template<class T, class Future>
  std::future<T> future_cast(Future& fut) const
  {
    function_called_ = true;
    return agency::future_traits<Future>::template cast<T>(fut);
  }

  mutable bool function_called_;
};


template<class Executor>
void test(Executor&& exec)
{
  // cast to a void future
  {
    auto from = agency::make_ready_future<int>(exec, 13);
    auto to = agency::future_cast<void>(exec, from);

    assert(!from.valid());
    assert(to.valid());

    to.wait();
  }

  // cast from an int to an unsigned int future
  {
    auto from = agency::make_ready_future<int>(exec, 13);
    auto to = agency::future_cast<unsigned int>(exec, from);

    assert(!from.valid());
    assert(to.valid());

    assert(to.get() == 13);
  }
}

int main()
{
  test(bulk_twoway_executor());
  test(bulk_then_executor());

  test(not_a_bulk_twoway_executor());
  test(not_a_bulk_then_executor());

  test(complete_bulk_executor());

  test(then_executor_with_future_cast());
  test(bulk_then_executor_with_future_cast());

  std::cout << "OK" << std::endl;

  return 0;
}

