#include <cassert>
#include <agency/cuda/future.hpp>
#include <iostream>

struct empty1 {};
struct empty2 {};

int main()
{
  using namespace agency::cuda;

  static_assert(agency::detail::is_future<deferred_future<int>>::value, "deferred_future<int> is not a future");

  {
    // default construction
    deferred_future<int> f0;
    assert(!f0.valid());
    assert(!f0.is_ready());
  }

  {
    // make_ready void
    deferred_future<void> f0 = deferred_future<void>::make_ready();
    assert(f0.valid());
    assert(f0.is_ready());
  }

  {
    // make_ready empty
    deferred_future<empty1> f0 = deferred_future<empty1>::make_ready();
    assert(f0.valid());
    assert(f0.is_ready());

    // move from empty to empty
    deferred_future<empty2> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
    assert(f1.is_ready());
  }

  {
    // move construction
    deferred_future<int> f0 = deferred_future<int>::make_ready(13);
    assert(f0.valid());
    assert(f0.is_ready());

    deferred_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(!f0.is_ready());
    assert(f1.valid());
    assert(f1.is_ready());
  }

  {
    // move assignment
    deferred_future<int> f1 = deferred_future<int>::make_ready(13);
    assert(f1.valid());
    assert(f1.is_ready());

    deferred_future<int> f2;
    assert(!f2.valid());
    assert(!f2.is_ready());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(!f1.is_ready());
    assert(f2.valid());
    assert(f2.is_ready());
  }

  {
    // make_ready/then
    auto f2 = deferred_future<int>::make_ready(13);

    assert(f2.valid());
    assert(f2.is_ready());

    auto f3 = f2.then([](int& arg)
    {
      return arg + 7;
    });

    assert(!f2.valid());
    assert(!f2.is_ready());
    assert(f3.valid());
    assert(!f3.is_ready());

    auto f4 = f3.then([](int& arg)
    {
      return arg + 42;
    });

    assert(!f3.valid());
    assert(!f3.is_ready());
    assert(f4.valid());
    assert(!f4.is_ready());

    f4.wait();
    assert(f4.valid());
    assert(f4.is_ready());

    assert(f4.get() == 13 + 7 + 42);

    assert(!f4.valid());
    assert(!f4.is_ready());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

