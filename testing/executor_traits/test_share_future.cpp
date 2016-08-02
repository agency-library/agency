#include <agency/agency.hpp>
#include <iostream>

int main()
{
  {
    // std::future<int> -> single std::shared_future<int> via executor_traits
    using executor_type = agency::concurrent_executor;
    using traits = agency::executor_traits<executor_type>;

    executor_type exec;
    auto f1 = traits::make_ready_future<int>(exec, 13);
    auto f2 = traits::share_future(exec, f1);

    assert(!f1.valid());
    assert(f2.valid());
    assert(f2.get() == 13);
  }

  {
    // std::future<int> -> multiple std::shared_future<int> via executor_traits + factory
    using executor_type = agency::concurrent_executor;
    using traits = agency::executor_traits<executor_type>;

    auto factory = [](traits::shape_type shape)
    {
      return std::vector<traits::shared_future<int>>(shape);
    };

    executor_type exec;
    auto f1 = traits::make_ready_future<int>(exec, 13);
    auto shared_futures = traits::share_future(exec, f1, factory, 10);

    assert(!f1.valid());

    for(auto& sf : shared_futures)
    {
      assert(sf.valid());
      assert(sf.get() == 13);
    }
  }

  {
    // std::future<int> -> multiple std::shared_future<int> via executor_traits
    using executor_type = agency::concurrent_executor;
    using traits = agency::executor_traits<executor_type>;

    executor_type exec;
    auto f1 = traits::make_ready_future<int>(exec, 13);
    auto shared_futures = traits::share_future(exec, f1, 10);

    assert(!f1.valid());

    for(auto& sf : shared_futures)
    {
      assert(sf.valid());
      assert(sf.get() == 13);
    }
  }

  std::cout << "OK" << std::endl;

  return 0;
}

