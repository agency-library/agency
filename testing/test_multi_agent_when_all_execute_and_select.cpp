#include <agency/future.hpp>
#include <agency/new_executor_traits.hpp>
#include <iostream>
#include <cassert>

struct my_executor {};

int main()
{
  {
    size_t n = 100;

    auto int_ready   = agency::detail::make_ready_future(0);
    auto void_ready  = agency::detail::make_ready_future();
    auto vector_ready = agency::detail::make_ready_future(std::vector<int>(n));

    auto futures = std::make_tuple(std::move(int_ready), std::move(void_ready), std::move(vector_ready));

    std::mutex mut;
    my_executor exec;
    std::future<agency::detail::tuple<std::vector<int>,int>> fut = agency::new_executor_traits<my_executor>::when_all_execute_and_select<2,0>(exec, [&mut](size_t idx, int& x, std::vector<int>& vec)
    {
      mut.lock();
      x += 1;
      mut.unlock();

      vec[idx] = 13;
    },
    n,
    std::move(futures));

    auto got = fut.get();

    assert(std::get<0>(got) == std::vector<int>(n, 13));
    assert(std::get<1>(got) == n);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

