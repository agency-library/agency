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
    // async_execute returning user-specified container
    
    my_executor exec;

    size_t n = 100;

    std::future<std::vector<int>> fut = agency::new_executor_traits<my_executor>::template async_execute<std::vector<int>>(exec, [](size_t idx)
    {
      return idx;
    },
    n);

    auto result = fut.get();

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

  {
    // async_execute returning default container
    
    my_executor exec;

    size_t n = 100;

    auto fut = agency::new_executor_traits<my_executor>::async_execute(exec, [](size_t idx)
    {
      return idx;
    },
    n);

    auto result = fut.get();

    std::vector<int> ref(n);
    std::iota(ref.begin(), ref.end(), 0);

    assert(std::equal(ref.begin(), ref.end(), result.begin()));
  }

//  {
//    // then_execute returning void
//    
//    my_executor exec;
//
//    size_t n = 100;
//
//    auto past = agency::detail::make_ready_future(13);
//
//    int increment_me = 0;
//    std::mutex mut;
//    auto fut = agency::new_executor_traits<my_executor>::then_execute(exec, past, [&](int& past, size_t idx)
//    {
//      mut.lock();
//      increment_me += past;
//      mut.unlock();
//    },
//    n);
//
//    fut.wait();
//
//    assert(increment_me == n * 13);
//  }

  std::cout << "OK" << std::endl;

  return 0;
}

