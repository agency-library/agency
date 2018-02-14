#include <cassert>
#include <agency/future/always_ready_future.hpp>
#include <agency/future/future_traits.hpp>
#include <iostream>

int main()
{
  using namespace agency;

  static_assert(agency::is_future<always_ready_future<int>>::value, "always_ready_future<int> is not a future");

  {
    // make_ready int
    always_ready_future<int> f0 = always_ready_future<int>::make_ready(13);
    assert(f0.valid());
    assert(f0.get() == 13);
  }

  {
    // make_ready void
    always_ready_future<void> f0 = always_ready_future<void>::make_ready();
    assert(f0.valid());
  }

  {
    // move construct int
    always_ready_future<int> f0 = always_ready_future<int>::make_ready(13);
    assert(f0.valid());

    always_ready_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
    assert(f1.get() == 13);
  }

  {
    // move construct void
    always_ready_future<void> f0 = always_ready_future<void>::make_ready();
    assert(f0.valid());

    always_ready_future<void> f1 = std::move(f0);
    assert(!f0.valid());
    assert(f1.valid());
  }

  {
    // move convert std::future<void>
    int set_me_to_thirteen = 0;
    std::future<void> f1 = std::async([&] { set_me_to_thirteen = 13; });
    assert(f1.valid());
    
    always_ready_future<void> f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(set_me_to_thirteen == 13);
  }

  {
    // move convert std::future<int>
    std::future<int> f1 = std::async([] { return 13; });
    
    always_ready_future<int> f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.get() == 13);
  }

  {
    // move assign int
    always_ready_future<int> f1 = always_ready_future<int>::make_ready(13);
    assert(f1.valid());

    always_ready_future<int> f2(7);
    assert(f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.get() == 13);
  }

  {
    // move assign void
    always_ready_future<void> f1 = always_ready_future<void>::make_ready();
    assert(f1.valid());

    always_ready_future<void> f2;
    assert(f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
  }

  {
    // move assign std::future<void>
    int set_me_to_thirteen = 0;
    std::future<void> f1 = std::async([&]{ set_me_to_thirteen = 13; });
    assert(f1.valid());

    always_ready_future<void> f2;
    assert(f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(set_me_to_thirteen == 13);
  }

  {
    // move assign std::future<int>
    std::future<int> f1 = std::async([]{ return 13; });
    assert(f1.valid());

    always_ready_future<int> f2;
    assert(!f2.valid());

    f2 = std::move(f1);
    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.get() == 13);
  }

  {
    // then int -> int
    auto f1 = always_ready_future<int>(7);

    bool continuation_executed = false;

    auto f2 = f1.then([&](int& x)
    {
      continuation_executed = true;
      return x + 13;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7 + 13);
  }

  {
    // then int -> void
    auto f1 = always_ready_future<int>(13);

    bool continuation_executed = false;

    auto f2 = f1.then([&](int &x)
    {
      continuation_executed = true;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());

    f2.get();
    assert(!f2.valid());
  }

  {
    // then void -> int
    always_ready_future<void> f1;

    bool continuation_executed = false;

    auto f2 = f1.then([&]()
    {
      continuation_executed = true;
      return 7;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());

    int result = f2.get();
    assert(!f2.valid());

    assert(result == 7);
  }

  {
    // then void -> void
    always_ready_future<void> f1;

    bool continuation_executed = false;

    auto f2 = f1.then([&]()
    {
      continuation_executed = true;
    });

    assert(continuation_executed);

    assert(!f1.valid());
    assert(f2.valid());

    f2.get();

    assert(!f2.valid());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

