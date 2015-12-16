#include <cassert>
#include "deferred_future.hpp"
#include <iostream>

int main()
{
  auto f1 = deferred_future<int>::make_ready(13);

  assert(f1.ready());
  assert(f1.valid());

  auto f2 = f1.then([](int& arg)
  {
    return arg + 7;
  });

  assert(!f1.valid());
  assert(!f2.ready());
  assert(f2.valid());

  auto f3 = f2.then([](int& arg)
  {
    return arg + 42;
  });

  assert(!f2.valid());
  assert(!f3.ready());
  assert(f3.valid());

  f3.wait();
  assert(f3.valid());
  assert(f3.ready());

  assert(f3.get() == 13 + 7 + 42);

  assert(!f3.valid());

  std::cout << "OK" << std::endl;

  return 0;
}

