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

    template<class Function, class ResultFactory, class SharedFactory>
    agency::detail::result_of_t<ResultFactory()>
      bulk_sync_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory)
    {
      auto result = result_factory();
      auto shared_parm = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_parm);
      }

      return std::move(result);
    }
};


} // end agency

