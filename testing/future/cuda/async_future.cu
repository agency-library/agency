#include <cassert>
#include <agency/cuda/future.hpp>
#include <iostream>

int main()
{
  using namespace agency::cuda;

  static_assert(agency::detail::is_future<async_future<int>>::value, "async_future<int> is not a future");

  {
    // default construction
    async_future<int> f0;
    assert(!f0.valid());
    assert(!f0.is_ready());
  }

  {
    // move construction
    async_future<int> f0 = make_ready_async_future(13);
    assert(f0.valid());

    async_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    // move assignment
    async_future<int> f1 = make_ready_async_future(13);
    assert(f1.valid());

    async_future<int> f2;
    assert(!f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // make_ready/then
    auto f2 = make_ready_async_future(13);

    assert(f2.is_ready());
    assert(f2.valid());

    auto f3 = f2.then([] __host__ __device__ (int& arg)
    {
      return arg + 7;
    });

    assert(!f2.valid());
    assert(f3.valid());

    auto f4 = f3.then([] __host__ __device__ (int& arg)
    {
      return arg + 42;
    });

    assert(!f3.valid());
    assert(f4.valid());

    f4.wait();
    assert(f4.valid());
    assert(f4.is_ready());

    assert(f4.get() == 13 + 7 + 42);

    assert(!f4.valid());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

