#include <agency/cuda/future.hpp>
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
    // empty -> void
    using future_type1 = agency::cuda::future<empty>;
    future_type1 f1 = agency::cuda::make_ready_future(empty());

    using future_type2 = agency::cuda::future<void>;
    future_type2 f2 = agency::future_traits<future_type1>::cast<void>(f1);
  }

//  {
//    using future_type1 = agency::cuda::future<unsigned int>;
//    future_type1 f1 = agency::cuda::make_ready_future(0u);
//
//    using future_type2 = agency::cuda::future<int>;
//    future_type2 f2 = agency::future_traits<future_type2>::cast(f1);
//  }   

  std::cout << "OK" << std::endl;

  return 0;
}

