#pragma once

#include <agency/future.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>

#include <thread>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>


namespace agency
{


class concurrent_executor
{
  public:
    using execution_category = concurrent_execution_tag;

    template<class Function, class Factory1, class T, class Factory2>
    std::future<detail::result_of_t<Factory1(size_t)>>
      then_execute(Function f, Factory1 result_factory, size_t n, std::future<T>& fut, Factory2 shared_factory)
    {
      return this->then_execute_impl(f, result_factory, n, fut, shared_factory);
    }

    template<class Function, class Factory1, class T, class Factory2>
    std::future<agency::detail::result_of_t<Factory1(size_t)>>
      then_execute(Function f, Factory1 result_factory, size_t n, std::shared_future<T>& fut, Factory2 shared_factory)
    {
      return this->then_execute_impl(f, result_factory, n, fut, shared_factory);
    }

    size_t unit_shape() const
    {
      constexpr size_t default_result = 1;
      size_t hw_concurrency = std::thread::hardware_concurrency();

      // hardware_concurency() is allowed to return 0, so guard against a 0 result
      return hw_concurrency ? hw_concurrency : default_result;
    }

    // XXX eliminate this when we eliminate executor_traits
    size_t shape() const
    {
      return unit_shape();
    }

    template<class Function, class Future, class ResultFactory, class SharedFactory>
    std::future<
      detail::result_of_t<ResultFactory()>
    >
    bulk_then_execute(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory)
    {
      return bulk_then_execute_impl(f, n, predecessor, result_factory, shared_factory);
    }

  private:
    template<class Function, class Factory1, class Future, class Factory2>
    std::future<agency::detail::result_of_t<Factory1(size_t)>>
      then_execute_impl(Function f, Factory1 result_factory, size_t n, Future& fut, Factory2 shared_factory,
                        typename std::enable_if<
                          !std::is_void<
                            typename future_traits<Future>::value_type
                          >::value
                        >::type* = 0)
    {
      if(n > 0)
      {
        using past_parameter_type = typename future_traits<Future>::value_type;

        return detail::monadic_then(fut, std::launch::async, [=](past_parameter_type& past_parameter) mutable
        {
          // put all the shared parameters on the first thread's stack
          auto result = result_factory(n);
          auto shared_parameter = shared_factory();

          // create a lambda to handle parameter passing
          auto g = [&,f](size_t idx)
          {
            result[idx] = agency::detail::invoke(f, idx, past_parameter, shared_parameter);
          };

          size_t mid = n / 2;

          std::future<void> left = detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(g, 0, mid);
          }

          std::future<void> right = detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(g, mid + 1, n);
          }

          g(mid);

          left.wait();
          right.wait();

          return std::move(result);
        });
      }

      return detail::make_ready_future(result_factory(n));
    }

    template<class Function, class Factory1, class Future, class Factory2>
    std::future<agency::detail::result_of_t<Factory1(size_t)>>
      then_execute_impl(Function f, Factory1 result_factory, size_t n, Future& fut, Factory2 shared_factory,
                        typename std::enable_if<
                          std::is_void<
                            typename future_traits<Future>::value_type
                          >::value
                        >::type* = 0)
    {
      if(n > 0)
      {
        return detail::monadic_then(fut, std::launch::async, [=]() mutable
        {
          // put all the shared parameters on the first thread's stack
          auto result = result_factory(n);
          auto shared_parameter = shared_factory();

          // create a lambda to handle parameter passing
          auto g = [&,f](size_t idx) mutable
          {
            result[idx] = agency::detail::invoke(f, idx, shared_parameter);
          };

          size_t mid = n / 2;

          std::future<void> left = detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(g, 0, mid);
          }

          std::future<void> right = detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(g, mid + 1, n);
          }

          g(mid);

          left.wait();
          right.wait();

          return std::move(result);
        });
      }

      return detail::make_ready_future(result_factory(n));
    }

  private:
    template<class Function, class Future, class ResultFactory, class SharedFactory>
    std::future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute_impl(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory,
                             typename std::enable_if<
                               !std::is_void<
                                 typename agency::future_traits<Future>::value_type
                               >::value
                             >::type* = 0)
    {
      if(n > 0)
      {
        using predecessor_type = typename agency::future_traits<Future>::value_type;

        return agency::detail::monadic_then(predecessor, std::launch::async, [=](predecessor_type& predecessor) mutable
        {
          // put all the shared parameters on the first thread's stack
          auto result = result_factory();
          auto shared_parameter = shared_factory();

          // create a lambda to handle parameter passing
          auto g = [&,f](size_t idx)
          {
            agency::detail::invoke(f, idx, predecessor, result, shared_parameter);
          };

          size_t mid = n / 2;

          std::future<void> left = agency::detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(g, 0, mid);
          }

          std::future<void> right = agency::detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(g, mid + 1, n);
          }

          g(mid);

          left.wait();
          right.wait();

          return std::move(result);
        });
      }

      return agency::detail::make_ready_future(result_factory());
    }

    template<class Function, class Future, class ResultFactory, class SharedFactory>
    std::future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute_impl(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory,
                             typename std::enable_if<
                               std::is_void<
                                 typename agency::future_traits<Future>::value_type
                               >::value
                             >::type* = 0)
    {
      if(n > 0)
      {
        return agency::detail::monadic_then(predecessor, std::launch::async, [=]() mutable
        {
          // put all the shared parameters on the first thread's stack
          auto result = result_factory();
          auto shared_parameter = shared_factory();

          // create a lambda to handle parameter passing
          auto g = [&,f](size_t idx)
          {
            agency::detail::invoke(f, idx, result, shared_parameter);
          };

          size_t mid = n / 2;

          std::future<void> left = agency::detail::make_ready_future();
          if(0 < mid)
          {
            left = this->async_execute(g, 0, mid);
          }

          std::future<void> right = agency::detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async_execute(g, mid + 1, n);
          }

          g(mid);

          left.wait();
          right.wait();

          return std::move(result);
        });
      }

      return agency::detail::make_ready_future(result_factory());
    }

    // first must be less than last
    template<class Function>
    std::future<void> async_execute(Function f, size_t first, size_t last)
    {
      return std::async(std::launch::async, [=]() mutable
      {
        size_t mid = (last + first) / 2;

        std::future<void> left = detail::make_ready_future();
        if(first < mid)
        {
          left = this->async_execute(f, first, mid);
        }

        std::future<void> right = detail::make_ready_future();
        if(mid + 1 < last)
        {
          right = this->async_execute(f, mid + 1, last);
        }

        agency::detail::invoke(f,mid);

        left.wait();
        right.wait();
      });
    }
};


} // end agency

