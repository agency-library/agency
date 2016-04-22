#include <agency/cuda/future.hpp>
#include <iostream>
#include <cassert>

struct return_void
{
  __device__
  void operator()()
  {
    printf("Called void -> void\n");
  }

  __device__
  void operator()(int x)
  {
    printf("Called int -> void\n");
    assert(x == 7);
  }
};

struct return_int
{
  __device__
  int operator()()
  {
    printf("Called void -> int\n");
    return 13;
  }

  __device__
  int operator()(int x)
  {
    printf("Called int -> int\n");
    assert(x == 7);

    return 13;
  }
};

int main()
{
  {
    // void -> void
    auto f1 = agency::cuda::make_ready_async_future();
    auto f2 = f1.then(return_void());

    assert(!f1.valid());
    f2.wait();
  }

  {
    // int -> void
    auto f1 = agency::cuda::make_ready_async_future(7);
    auto f2 = f1.then(return_void());

    assert(!f1.valid());
    f2.wait();
  }

  {
    // void -> int
    auto f1 = agency::cuda::make_ready_async_future();
    auto f2 = f1.then(return_int());

    assert(!f1.valid());
    assert(f2.get() == 13);
  }

  {
    // int -> int
    auto f1 = agency::cuda::make_ready_async_future(7);
    auto f2 = f1.then(return_int());

    assert(!f1.valid());
    assert(f2.get() == 13);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

