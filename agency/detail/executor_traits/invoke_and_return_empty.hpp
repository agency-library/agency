#pragma once

#include <agency/detail/config.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Function>
struct invoke_and_return_empty
{
  mutable Function f;

  struct empty {};

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  empty operator()(const Index& idx, Args&... args) const
  {
    f(idx, args...);

    // return something which can be cheaply discarded
    return empty();
  }
};


} // end new_executor_traits_detail
} // end detail
} // end agency

