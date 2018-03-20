#pragma once

#include <agency/future/always_ready_future.hpp>
#include <agency/execution/execution_categories.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>
#include <functional>
#include <utility>

namespace agency
{


class sequenced_executor
{
  public:
    using execution_category = sequenced_execution_tag;

    template<class T>
    using future = always_ready_future<T>;

    // XXX this shouldn't be necessary
    // always_blocking_t::static_query_v should be smart enough to determine from
    // the future type, and the absense of oneway functions,
    // that the executor is always blocking
    __AGENCY_ANNOTATION
    constexpr static bool query(always_blocking_t)
    {
      return true;
    }

    template<class Function, class ResultFactory, class SharedFactory>
    __AGENCY_ANNOTATION
    always_ready_future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      auto result = result_factory();
      auto shared_parm = shared_factory();

      for(size_t i = 0; i < n; ++i)
      {
        f(i, result, shared_parm);
      }

      return agency::make_always_ready_future(std::move(result));
    }

    __AGENCY_ANNOTATION
    friend constexpr bool operator==(const sequenced_executor&, const sequenced_executor&) noexcept
    {
      return true;
    }

    __AGENCY_ANNOTATION
    friend constexpr bool operator!=(const sequenced_executor&, const sequenced_executor&) noexcept
    {
      return false;
    }
};


} // end agency

