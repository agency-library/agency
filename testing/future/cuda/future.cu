#include <cassert>
#include <agency/cuda/future.hpp>
#include <iostream>

struct continuation_1
{
  __host__ __device__
  int operator()(int& arg)
  {
    return arg + 7;
  }
};

struct continuation_2
{
  __host__ __device__
  int operator()(int& arg)
  {
    return arg + 42;
  };
};

int main()
{
  using namespace agency::cuda;

  static_assert(agency::detail::is_future<future<int>>::value, "cuda::future<int> is not a future");

  {
    using shared_future_type = agency::future_traits<future<int>>::shared_future_type;
    using expected_shared_future_type = shared_future<int>;

    static_assert(
      std::is_same<
        shared_future_type,
        expected_shared_future_type
      >::value,
      "Unexpected associated shared_future type"
    );
  }

  {
    // default construction
    future<int> f0;
    assert(!f0.valid());
  }

  {
    // move construction
    future<int> f0 = future<int>::make_ready(13);
    assert(f0.valid());

    future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    // move assignment
    future<int> f1 = future<int>::make_ready(13);
    assert(f1.valid());

    future<int> f2;
    assert(!f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // make_ready/then
    auto f1 = future<int>::make_ready(13);

    assert(f1.valid());

    auto f2 = f1.then(continuation_1());

    assert(!f1.valid());
    assert(f2.valid());

    auto f3 = f2.then(continuation_2());

    assert(!f2.valid());
    assert(f3.valid());

    f3.wait();
    assert(f3.valid());

    assert(f3.get() == 13 + 7 + 42);

    assert(!f3.valid());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

