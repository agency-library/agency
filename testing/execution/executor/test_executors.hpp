#pragma once

#include <type_traits>
#include <agency/detail/invoke.hpp>
#include <agency/future.hpp>
#include <future>

// this type falls into no executor categories
struct not_an_executor {};


class then_executor
{
  public:
    template<class Function, class T>
    std::future<agency::detail::result_of_t<Function(T&)>>
      then_execute(Function&& f, std::future<T>& predecessor) const
    {
      return std::async(std::launch::async, [](std::future<T>&& predecessor, Function&& f)
      {
        T arg = predecessor.get();
        return std::forward<Function>(f)(arg);
      },
      std::move(predecessor),
      std::forward<Function>(f)
      );
    }

    template<class Function>
    std::future<agency::detail::result_of_t<Function()>>
      then_execute(Function&& f, std::future<void>& predecessor) const
    {
      return std::async(std::launch::async, [](std::future<void>&& predecessor, Function&& f)
      {
        predecessor.get();
        return std::forward<Function>(f)();
      },
      std::move(predecessor),
      std::forward<Function>(f)
      );
    }
};


class twoway_executor
{
  public:
    template<class Function>
    std::future<agency::detail::result_of_t<Function()>>
      twoway_execute(Function&& f) const
    {
      return std::async(std::launch::async, std::forward<Function>(f));
    }
};


class synchronous_executor
{
  public:
    template<class Function>
    agency::detail::result_of_t<Function()>
      sync_execute(Function&& f) const
    {
      return std::forward<Function>(f)();
    }
};


class bulk_then_executor
{
  public:
    template<class Function, class Future, class ResultFactory, class SharedFactory>
    std::future<
      typename std::result_of<ResultFactory()>::type
    >
    bulk_then_execute(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      return bulk_then_execute_impl(f, n, predecessor, result_factory, shared_factory);
    }

  private:
    template<class Function, class Future, class ResultFactory, class SharedFactory>
    std::future<agency::detail::result_of_t<ResultFactory()>>
      bulk_then_execute_impl(Function f, size_t n, Future& predecessor, ResultFactory result_factory, SharedFactory shared_factory,
                             typename std::enable_if<
                               !std::is_void<
                                 agency::future_result_t<Future>
                               >::value
                             >::type* = 0) const
    {
      if(n > 0)
      {
        using predecessor_type = agency::future_result_t<Future>;

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
            left = this->async(g, 0, mid);
          }

          std::future<void> right = agency::detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async(g, mid + 1, n);
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
                                 agency::future_result_t<Future>
                               >::value
                             >::type* = 0) const
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
            left = this->async(g, 0, mid);
          }

          std::future<void> right = agency::detail::make_ready_future();
          if(mid + 1 < n)
          {
            right = this->async(g, mid + 1, n);
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
    std::future<void> async(Function f, size_t first, size_t last) const
    {
      return std::async(std::launch::async, [=]() mutable
      {
        size_t mid = (last + first) / 2;

        std::future<void> left = agency::detail::make_ready_future();
        if(first < mid)
        {
          left = this->async(f, first, mid);
        }

        std::future<void> right = agency::detail::make_ready_future();
        if(mid + 1 < last)
        {
          right = this->async(f, mid + 1, last);
        }

        agency::detail::invoke(f,mid);

        left.wait();
        right.wait();
      });
    }
};


class bulk_twoway_executor
{
  public:
    template<class Function, class ResultFactory, class SharedFactory>
    std::future<
      typename std::result_of<ResultFactory()>::type
    >
    bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      return std::async(std::launch::async, [=]
      {
        auto result = result_factory();
        auto shared_parm = shared_factory();

        for(size_t i = 0; i < n; ++i)
        {
          f(i, result, shared_parm);
        }

        return std::move(result);
      });
    }
};


// these executor types fall into one category
struct not_a_bulk_twoway_executor : bulk_then_executor {};
struct not_a_bulk_then_executor : bulk_twoway_executor {};


// this executor type falls into two categories
struct complete_bulk_executor : bulk_twoway_executor, bulk_then_executor {};


struct bulk_executor_without_shape_type
{
  template<class Function, class ResultFactory, class SharedFactory>
  std::future<typename std::result_of<ResultFactory()>::type>
  bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const;
};

struct bulk_executor_with_shape_type
{
  struct shape_type
  {
    size_t n;
  };

  template<class Function, class ResultFactory, class SharedFactory>
  std::future<typename std::result_of<ResultFactory()>::type>
  bulk_twoway_execute(Function f, shape_type n, ResultFactory result_factory, SharedFactory shared_factory) const;
};

