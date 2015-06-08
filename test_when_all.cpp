#include <agency/future.hpp>
#include <iostream>
#include <cassert>

int main()
{
  {
    auto void_future1 = agency::when_all();

    auto void_future2 = agency::when_all(void_future1);

    void_future2.get();
  }

  {
    auto int_ready = agency::detail::make_ready_future(13);
    auto float_ready = agency::detail::make_ready_future(7.f);

    std::future<agency::detail::tuple<int,float>> fut = agency::when_all(int_ready, float_ready);

    auto got = fut.get();

    assert(std::get<0>(got) == 13);
    assert(std::get<1>(got) == 7.f);
  }

  {
    auto int_ready   = agency::detail::make_ready_future(13);
    auto void_ready  = agency::detail::make_ready_future();
    auto float_ready = agency::detail::make_ready_future(7.f);

    std::future<agency::detail::tuple<int,float>> fut = agency::when_all(int_ready, void_ready, float_ready);

    auto got = fut.get();

    assert(std::get<0>(got) == 13);
    assert(std::get<1>(got) == 7.f);
  }

  {
    auto int_ready   = agency::detail::make_ready_future(13);
    auto void_ready  = agency::detail::make_ready_future();
    auto float_ready = agency::detail::make_ready_future(7.f);

    std::future<int> fut = agency::when_all_and_select<0>(int_ready, void_ready, float_ready);

    auto got = fut.get();

    assert(got == 13);
  }

  {
    auto int_ready   = agency::detail::make_ready_future(13);
    auto void_ready  = agency::detail::make_ready_future();
    auto float_ready = agency::detail::make_ready_future(7.f);

    std::future<agency::detail::tuple<float,int>> fut = agency::when_all_and_select<2,0>(int_ready, void_ready, float_ready);

    auto got = fut.get();

    assert(std::get<0>(got) == 7.f);
    assert(std::get<1>(got) == 13);
  }

  {
    auto int_ready   = agency::detail::make_ready_future(13);
    auto void_ready  = agency::detail::make_ready_future();
    auto float_ready = agency::detail::make_ready_future(7.f);

    auto futures = std::make_tuple(std::move(int_ready), std::move(void_ready), std::move(float_ready));

    std::future<agency::detail::tuple<float,int>> fut = agency::when_all_execute_and_select<2,0>(std::move(futures), [](int& x, float& y)
    {
      x += 1;
      y += 2;
    });

    auto got = fut.get();

    assert(std::get<0>(got) == 9.f);
    assert(std::get<1>(got) == 14);
  }

  std::cout << "OK" << std::endl;

  return 0;
}

