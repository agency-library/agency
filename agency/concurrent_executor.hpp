#pragma once

#include <thread>
#include <vector>
#include <memory>
#include <agency/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/functional.hpp>
#include <algorithm>


namespace agency
{


class concurrent_executor
{
  public:
    using execution_category = concurrent_execution_tag;

    // XXX eliminate this after we implement executor_traits::make_ready_future()'s default behavior
    inline std::future<void> make_ready_future()
    {
      return detail::make_ready_future();
    }

    template<class Function, class T>
    std::future<void> then_execute(std::future<void>& fut, Function f, size_t n, T&& shared_init)
    {
      std::future<void> result = make_ready_future();

      if(n > 0)
      {
        result = detail::then(fut, std::launch::async, [=](std::future<void>&) mutable
        {
          // put the shared parameter on the first thread's stack
          auto shared_parm = agency::decay_construct(shared_init);

          size_t mid = n / 2;

          std::future<void> left = make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(f, 0, mid, shared_parm);
          }

          std::future<void> right = make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(f, mid + 1, n, shared_parm);
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

        std::future<void> left = make_ready_future();
        if(first < mid)
        {
          left = this->async_execute(f, first, mid, shared_parm);
        }

        std::future<void> right = make_ready_future();
        if(mid + 1 < last)
        {
          right = this->async_execute(f, mid + 1, last, shared_parm);
        }

        f(mid, shared_parm);

        left.wait();
        right.wait();
      });
    }
};


} // end agency

