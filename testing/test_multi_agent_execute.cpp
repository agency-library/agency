#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <mutex>

struct my_executor {};

int main()
{
  {
    // execute returning user-specified container
    
    my_executor exec;

    size_t n = 100;

    std::vector<int> result = agency::new_executor_traits<my_executor>::template execute<std::vector<int>>(exec, [](size_t idx)
    {
      return idx;
    },
    n);

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

  {
    // execute returning default container
    
    my_executor exec;

    size_t n = 100;

    auto result = agency::new_executor_traits<my_executor>::execute(exec, [](size_t idx)
    {
      return idx;
    },
    n);

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

  {
    // execute returning void
    
    my_executor exec;

    size_t n = 100;

    int increment_me = 0;
    std::mutex mut;
    agency::new_executor_traits<my_executor>::execute(exec, [&](size_t idx)
    {
      mut.lock();
      increment_me += 13;
      mut.unlock();
    },
    n);

    assert(increment_me == n * 13);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

