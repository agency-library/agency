#pragma once

#include <agency/future.hpp>
#include <agency/execution/execution_categories.hpp>
#include <functional>
#include <utility>

namespace agency
{


class sequenced_executor
{
  public:
    using execution_category = sequenced_execution_tag;

    template<class Function, class Factory>
    __AGENCY_ANNOTATION
    void execute(Function f, size_t n, Factory shared_factory)
    {
      auto shared_parm = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, shared_parm);
      }
    }

    template<class Function, class Factory>
    std::future<void> async_execute(Function f, size_t n, Factory shared_factory)
    {
      return std::async(std::launch::deferred, [=]
      {
        this->execute(f, n, shared_factory);
      });
    }

    template<class Function, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<void>& fut, Factory shared_factory)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<void>& predecessor)
      {
        this->execute(f, n, shared_factory);
      });
    }

    template<class Function, class T, class Factory>
    std::future<void> then_execute(Function f, size_t n, std::future<T>& fut, Factory shared_factory)
    {
      return detail::then(fut, std::launch::deferred, [=](std::future<T>& predecessor) mutable
      {
        using second_type = decltype(shared_factory());

        auto first = predecessor.get();

        this->execute([=](size_t idx, std::pair<T,second_type>& p) mutable
        {
          f(idx, p.first, p.second);
        },
        n,
        [=]() mutable
        {
          return std::make_pair(first, shared_factory());
        });
      });
    }
};


} // end agency

