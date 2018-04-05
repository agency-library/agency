#pragma once

#include <agency/future/always_ready_future.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>
#include <agency/execution/executor/properties/bulk_guarantee.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>

namespace agency
{
namespace experimental
{
namespace detail
{


template<size_t first, size_t last>
struct static_for_loop_impl
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&& f)
  {
    std::forward<Function>(f)(first);

    static_for_loop_impl<first+1,last>::invoke(std::forward<Function>(f));
  }
};


template<size_t first>
struct static_for_loop_impl<first,first>
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&&)
  {
  }
};


template<size_t n, class Function>
__AGENCY_ANNOTATION
void static_for_loop(Function&& f)
{
  static_for_loop_impl<0,n>::invoke(std::forward<Function>(f));
}


} // end detail


// XXX the type of unroll_factor should be auto
//     maybe it should be 32b for now?
template<std::size_t unroll_factor_>
class unrolling_executor
{
  public:
    __AGENCY_ANNOTATION
    constexpr static bulk_guarantee_t::sequenced_t query(bulk_guarantee_t)
    {
      return bulk_guarantee_t::sequenced_t();
    }

    // XXX this overload shouldn't be necessary
    // always_blocking_t::static_query_v should be smart enough to determine from
    // the future type, and the absense of oneway functions,
    // that this executor is always blocking
    __AGENCY_ANNOTATION
    constexpr static bool query(always_blocking_t)
    {
      return true;
    }

    static constexpr std::size_t unroll_factor = unroll_factor_;

    __AGENCY_ANNOTATION
    static constexpr std::size_t unit_shape()
    {
      return unroll_factor;
    }

    template<class T>
    using future = always_ready_future<T>;

    template<class Function, class ResultFactory, class SharedFactory>
    __AGENCY_ANNOTATION
    future<agency::detail::result_of_t<ResultFactory()>>
      bulk_twoway_execute(Function f, size_t n, ResultFactory result_factory, SharedFactory shared_factory) const
    {
      auto result = result_factory();
      auto shared_parm = shared_factory();

      // the following implementation is equivalent to this loop
      // however, #pragma unroll is not portable and
      // is not guaranteed to unroll anyway
      //#pragma unroll(unroll_factor)
      //for(size_t i = 0; i < n; ++i)
      //{
      //  f(i, result, shared_parm);
      //}

      // technically, these first two branches are not required for correctness
      // they're included because in these cases we can use the static_for_loop's
      // loop variable i directly without having to introduce an additional variable
      if(n == unroll_factor)
      {
        detail::static_for_loop<unroll_factor>([&](std::size_t i)
        {
          f(i, result, shared_parm);
        });
      }
      else if(n < unroll_factor)
      {
        detail::static_for_loop<unroll_factor>([&](std::size_t i)
        {
          if(i < n)
          {
            f(i, result, shared_parm);
          }
        });
      }
      else
      {
        std::size_t i = 0;

        // while the unroll_factor is no larger than the remaining work,
        // we don't need to guard the invocation of f()
        while(unroll_factor <= n - i)
        {
          detail::static_for_loop<unroll_factor>([&](std::size_t)
          {
            f(i, result, shared_parm);
            ++i;
          });
        }

        // the final loop is larger than the remaining work,
        // so we need to guard the invocation of f()
        if(n - i)
        {
          detail::static_for_loop<unroll_factor>([&](std::size_t)
          {
            if(i < n)
            {
              f(i, result, shared_parm);
              ++i;
            }
          });
        }
      }

      return agency::make_always_ready_future(std::move(result));
    }
};


} // end experimental
} // end agency

