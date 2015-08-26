#include <agency/cuda/future.hpp>
#include <agency/cuda/grid_executor.hpp>
#include <agency/future.hpp>
#include <iostream>
#include <cassert>

struct empty {};

int main()
{
  {
    // construction of future<void> from empty types

    using future_type1 = agency::cuda::future<empty>;
    future_type1 f1 = agency::cuda::make_ready_future(empty());

    using future_type2 = agency::cuda::future<void>;
    future_type2 f2(std::move(f1));

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // empty -> void via future_traits
    using future_type1 = agency::cuda::future<empty>;
    future_type1 f1 = agency::cuda::make_ready_future(empty());

    using future_type2 = agency::cuda::future<void>;
    future_type2 f2 = agency::future_traits<future_type1>::cast<void>(f1);

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // empty -> void via executor_traits
    using future_type1 = agency::cuda::future<empty>;
    future_type1 f1 = agency::cuda::make_ready_future(empty());

    using future_type2 = agency::cuda::future<void>;

    agency::cuda::grid_executor exec;

    future_type2 f2 = agency::executor_traits<agency::cuda::grid_executor>::future_cast<void>(exec, f1);

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // unsigned int -> int via future_traits
    using future_type1 = agency::cuda::future<unsigned int>;
    future_type1 f1 = agency::cuda::make_ready_future(13u);

    using future_type2 = agency::cuda::future<int>;
    future_type2 f2 = agency::future_traits<future_type1>::cast<int>(f1);

    // XXX fut.then() needs to invalidate fut
    //assert(!f1.valid());
    assert(f2.get() == 13);
  }   

  {
    // unsigned int -> int via executor_traits
    using future_type1 = agency::cuda::future<unsigned int>;
    future_type1 f1 = agency::cuda::make_ready_future(13u);

    using future_type2 = agency::cuda::future<int>;

    agency::cuda::grid_executor exec;

    future_type2 f2 = agency::executor_traits<agency::cuda::grid_executor>::future_cast<int>(exec, f1);

    // XXX fut.then() needs to invalidate fut
    //assert(!f1.valid());
    assert(f2.get() == 13);
  }   

  std::cout << "OK" << std::endl;

  return 0;
}

