#include <cassert>
#include <iostream>
#include <agency/cuda/future.hpp>

int main()
{
  using namespace agency::cuda;

  static_assert(agency::detail::is_future<shared_future<int>>::value, "shared_future<int> is not a future");

  {
    // default construction
    shared_future<int> f0;
    assert(!f0.valid());
    assert(!f0.is_ready());
  }

  {
    // move construction
    shared_future<int> f0 = shared_future<int>::make_ready(13);
    assert(f0.valid());
    assert(f0.is_ready());

    shared_future<int> f1 = std::move(f0);
    assert(!f0.valid());
    assert(!f0.is_ready());
    assert(f1.valid());
    assert(f1.is_ready());
  }

  {
    // move assignment
    shared_future<int> f1 = shared_future<int>::make_ready(13);
    assert(f1.valid());
    assert(f1.is_ready());

    shared_future<int> f2;
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
    auto f2 = shared_future<int>::make_ready(13);

    assert(f2.valid());
    assert(f2.is_ready());

    try
    {
      auto f3 = f2.then([] __host__ __device__ (int& arg)
      {
        return arg + 7;
      });

      // XXX shared_future.then() is unimplemented, so we should not have gotten to here
      assert(0);

      assert(f2.valid()); // f2 is a shared_future and should still be valid after a .then()
      assert(f3.valid()); // f3 is a future and should be valid

      auto f4 = f3.then([] __host__ __device__ (int& arg)
      {
        return arg + 42;
      });

      assert(!f3.valid()); // f3 is a future and should be invalid after a .then()
      assert(f4.valid());  // f4 should be valid

      f4.wait();
      assert(f4.valid());
      assert(f4.is_ready());

      shared_future<int> f5 = f4.share();
      assert(!f4.valid());
      assert(!f4.is_ready());
      assert(f5.valid());
      assert(f5.is_ready());

      assert(f5.get() == 13 + 7 + 42);

      assert(f5.valid()); // f5 is a shared_future and should be valid after a .get()
      assert(f5.is_ready()); // f5 is a shared_future and should still be ready after a .get()
    }
    catch(std::runtime_error)
    {
    }
  }

  std::cout << "OK" << std::endl;

  return 0;
}

