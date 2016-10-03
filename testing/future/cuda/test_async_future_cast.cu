#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <iostream>
#include <cassert>

struct empty {};

int main()
{
  {
    // construction of future<void> from empty types

    using future_type1 = agency::cuda::async_future<empty>;
    future_type1 f1 = agency::cuda::make_ready_async_future(empty());

    using future_type2 = agency::cuda::async_future<void>;
    future_type2 f2(std::move(f1));

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // empty -> void via future_traits
    using future_type1 = agency::cuda::async_future<empty>;
    future_type1 f1 = agency::cuda::make_ready_async_future(empty());

    using future_type2 = agency::cuda::async_future<void>;
    future_type2 f2 = agency::future_traits<future_type1>::cast<void>(f1);

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // empty -> void via future_cast()
    using future_type1 = agency::cuda::async_future<empty>;
    future_type1 f1 = agency::cuda::make_ready_async_future(empty());

    using future_type2 = agency::cuda::async_future<void>;

    agency::cuda::grid_executor exec;

    future_type2 f2 = agency::future_cast<void>(exec, f1);

    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // unsigned int -> int via future_traits
    using future_type1 = agency::cuda::async_future<unsigned int>;
    future_type1 f1 = agency::cuda::make_ready_async_future(13u);

    using future_type2 = agency::cuda::async_future<int>;
    future_type2 f2 = agency::future_traits<future_type1>::cast<int>(f1);

    // XXX fut.then() needs to invalidate fut
    //assert(!f1.valid());
    assert(f2.get() == 13);
  }   

  {
    // unsigned int -> int via future_cast()
    using future_type1 = agency::cuda::async_future<unsigned int>;
    future_type1 f1 = agency::cuda::make_ready_async_future(13u);

    using future_type2 = agency::cuda::async_future<int>;

    agency::cuda::grid_executor exec;

    future_type2 f2 = agency::future_cast<int>(exec, f1);

    // XXX fut.then() needs to invalidate fut
    //assert(!f1.valid());
    assert(f2.get() == 13);
  }   

  std::cout << "OK" << std::endl;

  return 0;
}

