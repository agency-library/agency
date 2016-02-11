#pragma once

#include <agency/detail/config.hpp>
#include <agency/functional.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
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
    agency::invoke(f, idx, args...);

    // return something which can be cheaply discarded
    return empty();
  }
};


} // end executor_traits_detail
} // end detail
} // end agency

