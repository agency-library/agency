#include <agency/cuda/future.hpp>
#include <cassert>

int main()
{
  {
    // int, int -> (int, int)
    auto f1 = agency::cuda::make_ready_future<int>(7);
    auto f2 = agency::cuda::make_ready_future<int>(13);

    auto f3 = agency::cuda::when_all(f1, f2);

    auto result = f3.get();

    assert(result == agency::detail::make_tuple(7,13));
    assert(!f1.valid());
    assert(!f2.valid());
  }

  {
    // int, void -> int
    auto f1 = agency::cuda::make_ready_future<int>(7);
    auto f2 = agency::cuda::make_ready_future();

    auto f3 = agency::cuda::when_all(f1, f2);

    auto result = f3.get();

    assert(result == 7);
    assert(!f1.valid());
    assert(!f2.valid());
  }

  {
    // void, int -> int
    auto f1 = agency::cuda::make_ready_future();
    auto f2 = agency::cuda::make_ready_future<int>(7);

    auto f3 = agency::cuda::when_all(f1, f2);

    auto result = f3.get();

    assert(result == 7);
    assert(!f1.valid());
    assert(!f2.valid());
  }

  {
    // int -> int
    auto f1 = agency::cuda::make_ready_future<int>(7);

    auto f2 = agency::cuda::when_all(f1);

    auto result = f2.get();

    assert(result == 7);
    assert(!f1.valid());
  }

  {
    // void -> void
    auto f1 = agency::cuda::make_ready_future();

    auto f2 = agency::cuda::when_all(f1);

    f2.wait();

    assert(!f1.valid());
  }

  std::cout << "OK" << std::endl;

  return 0;
}

