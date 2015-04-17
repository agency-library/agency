#pragma once

#include <agency/future.hpp>
#include <agency/execution_categories.hpp>
#include <agency/functional.hpp>
#include <functional>

namespace agency
{


class sequential_executor
{
  public:
    using execution_category = sequential_execution_tag;

    template<class Function, class T>
    void execute(Function f, size_t n, T&& shared_init)
    {
      auto shared_parm = agency::decay_construct(std::forward<T>(shared_init));

      for(size_t i = 0; i < n; ++i)
      {
        f(i, shared_parm);
      }
    }

    template<class Function, class T>
    std::future<void> async_execute(Function f, size_t n, T&& shared_init)
    {
      return std::async(std::launch::deferred, [=]
      {
        this->execute(f, n, shared_init);
      });
    }

    template<class Function, class T>
    std::future<void> then_execute(std::future<void>& fut, Function f, size_t n, T&& shared_init)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<void>& predecessor)
      {
        this->execute(f, n, shared_init);
      });
    }
};


} // end agency

