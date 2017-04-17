#include <cassert>
#include <agency/future/always_ready_future.hpp>
#include <agency/future/variant_future.hpp>
#include <agency/future.hpp>
#include <iostream>

template<class T>
using future = agency::variant_future<agency::always_ready_future<T>, std::future<T>>;

int main()
{
  static_assert(agency::is_future<future<int>>::value, "variant_future is not a future");

  {
    // make_ready int
    future<int> f0 = future<int>::make_ready(13);
    assert(f0.valid());
    assert(f0.get() == 13);
  }

  {
    // make_ready void
    future<void> f0 = future<void>::make_ready();
    assert(f0.valid());
  }

  {
    // move construct int
    future<int> f0 = future<int>::make_ready(13);
    assert(f0.valid());

    future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
    assert(f1.get() == 13);
  }

  {
    // move construct void
    future<void> f0 = future<void>::make_ready();
    assert(f0.valid());

    future<void> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    // move assign int
    future<int> f1 = agency::always_ready_future<int>::make_ready(13);
    assert(f1.valid());
    assert(f1.index() == 0);

    future<int> f2 = agency::future_traits<std::future<int>>::template make_ready<int>(7);
    assert(f2.valid());
    assert(f2.index() == 1);

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);
    assert(f2.get() == 13);
  }

  {
    // move assign void
    future<void> f1 = agency::always_ready_future<void>::make_ready();
    assert(f1.valid());
    assert(f1.index() == 0);

    future<void> f2 = agency::future_traits<std::future<int>>::make_ready();
    assert(f2.valid());
    assert(f2.index() == 1);

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);
  }

  {
    // then always_ready_future<int> -> always_ready_future<int>
    future<int> f1 = agency::always_ready_future<int>(7);

    bool continuation_executed = false;

    future<int> f2 = f1.then([&](int& x)
    {
      continuation_executed = true;
      return x + 13;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7 + 13);
  }

  {
    // then std::future<int> -> std::future<int>
    future<int> f1 = agency::future_traits<std::future<int>>::template make_ready<int>(7);

    future<int> f2 = f1.then([&](int& x)
    {
      return x + 13;
    });

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 1);

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7 + 13);
  }

  {
    // then always_ready_future<int> -> always_ready_future<void>
    future<int> f1 = agency::always_ready_future<int>(13);

    bool continuation_executed = false;

    future<void> f2 = f1.then([&](int &x)
    {
      continuation_executed = true;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);

    f2.get();
    assert(!f2.valid());
  }

  {
    // then std::future<int> -> std::future<void>
    future<int> f1 = agency::future_traits<std::future<int>>::template make_ready<int>(7);

    bool continuation_executed = false;

    future<void> f2 = f1.then([&](int &x)
    {
      continuation_executed = true;
    });

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 1);

    f2.get();

    assert(continuation_executed);
    assert(!f2.valid());
  }

  {
    // then always_ready_future<void> -> always_ready_future<int>
    future<void> f1 = agency::always_ready_future<void>();

    bool continuation_executed = false;

    future<int> f2 = f1.then([&]()
    {
      continuation_executed = true;
      return 7;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7);
  }

  {
    // then std::future<void> -> std::future<int>
    future<void> f1 = agency::future_traits<std::future<void>>::template make_ready<void>();

    future<int> f2 = f1.then([&]()
    {
      return 7;
    });

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 1);

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7);
  }

  {
    // then always_ready_future<void> -> always_ready_future<void>
    future<void> f1 = agency::always_ready_future<void>();

    bool continuation_executed = false;

    future<void> f2 = f1.then([&]()
    {
      continuation_executed = true;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 0);

    f2.get();

    assert(!f2.valid());
  }

  {
    // then std::future<void> -> std::future<void>
    future<void> f1 = agency::future_traits<std::future<void>>::template make_ready<void>();

    bool continuation_executed = false;

    future<void> f2 = f1.then([&]()
    {
      continuation_executed = true;
    });

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.index() == 1);

    f2.get();

    assert(continuation_executed);
    assert(!f2.valid());
  }

  std::cout << "OK" << std::endl;

  return 0;
}


