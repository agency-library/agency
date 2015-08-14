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

    template<class Function, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<void>& fut, Factory shared_factory)
    {
      std::future<void> result = detail::make_ready_future();

      if(n > 0)
      {
        result = detail::then(fut, std::launch::async, [=](std::future<void>&) mutable
        {
          // put the shared parameter on the first thread's stack
          auto shared_parm = shared_factory();

          size_t mid = n / 2;

          std::future<void> left = detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(f, 0, mid, shared_parm);
          }

          std::future<void> right = detail::make_ready_future();
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


    template<class Function, class T, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<T>& fut, Factory shared_factory)
    {
      std::future<void> result = detail::make_ready_future();

      if(n > 0)
      {
        result = detail::then(fut, std::launch::async, [=](std::future<T>& fut) mutable
        {
          T past_parameter = fut.get();

          // put the shared parameter on the first thread's stack
          auto shared_parm = shared_factory();

          size_t mid = n / 2;

          std::future<void> left = detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(f, 0, mid, past_parameter, shared_parm);
          }

          std::future<void> right = detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(f, mid + 1, n, past_parameter, shared_parm);
          }

          f(mid, past_parameter, shared_parm);

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
          left = this->async_execute(f, first, mid, shared_parm);
        }

        std::future<void> right = detail::make_ready_future();
        if(mid + 1 < last)
        {
          right = this->async_execute(f, mid + 1, last, shared_parm);
        }

        f(mid, shared_parm);

        left.wait();
        right.wait();
      });
    }

    // first must be less than last
    template<class Function, class T1, class T2>
    std::future<void> async_execute(Function f, size_t first, size_t last, T1& shared_parm1, T2& shared_parm2)
    {
      return std::async(std::launch::async, [=,&shared_parm1,&shared_parm2]() mutable
      {
        size_t mid = (last + first) / 2;

        std::future<void> left = detail::make_ready_future();
        if(first < mid)
        {
          left = this->async_execute(f, first, mid, shared_parm1, shared_parm2);
        }

        std::future<void> right = detail::make_ready_future();
        if(mid + 1 < last)
        {
          right = this->async_execute(f, mid + 1, last, shared_parm1, shared_parm2);
        }

        f(mid, shared_parm1, shared_parm2);

        left.wait();
        right.wait();
      });
    }
};


} // end agency

