#pragma once

#include <thread>
#include <vector>
#include <memory>
#include <agency/detail/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/functional.hpp>
#include <algorithm>


namespace agency
{


class concurrent_executor
{
  public:
    using execution_category = concurrent_execution_tag;

    template<class Function, class T>
    std::future<void> async_execute(Function f, size_t n, T shared_init)
    {
      auto shared_parm = agency::decay_construct(shared_init);

      std::future<void> result = detail::make_ready_future();

      if(n > 0)
      {
        result = std::async(std::launch::async, [=]() mutable
        {
          size_t mid = n / 2;

          std::future<void> left = detail::make_ready_future();
          if(0 < mid)
          {
            left = std::move(this->async_execute(f, 0, mid, shared_parm));
          }

          std::future<void> right = detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = std::move(this->async_execute(f, mid + 1, n, shared_parm));
          }

          f(mid, shared_parm);

          left.wait();
          right.wait();
        });
      }

      return result;
    }

  private:
    // first must be less than last
    template<class Function, class T>
    std::future<void> async_execute(Function f, size_t first, size_t last, T& shared_parm)
    {
      return std::async(std::launch::async, [=,&shared_parm]() mutable
      {
        size_t mid = (last + first) / 2;

        std::future<void> left = detail::make_ready_future();
        if(first < mid)
        {
          left = std::move(this->async_execute(f, first, mid, shared_parm));
        }

        std::future<void> right = detail::make_ready_future();
        if(mid + 1 < last)
        {
          right = std::move(this->async_execute(f, mid + 1, last, shared_parm));
        }

        f(mid, shared_parm);

        left.wait();
        right.wait();
      });
    }
};


} // end agency

