#include <cassert>
#include "deferred_future.hpp"
#include <iostream>

int main()
{
  deferred_future<int> f0;
  assert(!f0.valid());

  deferred_future<int> f1 = std::move(f0);
  assert(!f0.valid());
  assert(!f1.valid());

  auto f2 = deferred_future<int>::make_ready(13);

  assert(f2.ready());
  assert(f2.valid());

  auto f3 = f2.then([](int& arg)
  {
    return arg + 7;
  });

  assert(!f2.valid());
  assert(!f3.ready());
  assert(f3.valid());

  auto f4 = f3.then([](int& arg)
  {
    return arg + 42;
  });

  assert(!f3.valid());
  assert(!f4.ready());
  assert(f4.valid());

  f4.wait();
  assert(f4.valid());
  assert(f4.ready());

  assert(f4.get() == 13 + 7 + 42);

  assert(!f4.valid());

  std::cout << "OK" << std::endl;

  return 0;
}

