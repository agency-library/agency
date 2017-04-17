#pragma once

#include <agency/detail/config.hpp>
#include <agency/future/future_traits/future_rebind_value.hpp>
#include <agency/future.hpp>

namespace agency
{


template<class Future, class Function>
struct future_then_result
{
  // get the result of Function when invoked with Future's value
  using result_of_function = detail::result_of_continuation_t<Function,Future>;

  // lift the result into a Future of the appropriate type
  using type = future_rebind_value_t<Future, result_of_function>;
};


template<class Future, class Function>
using future_then_result_t = typename future_then_result<Future,Function>::type;


} // end agency

