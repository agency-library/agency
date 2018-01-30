#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/adaptors/twoway_executor.hpp>


namespace agency
{
namespace detail
{


template<class E, class Function>
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<detail::decay_t<Function>()>
>
twoway_execute(const E& exec, Function&& f)
{
  return detail::twoway_executor<E>(exec).twoway_execute(std::forward<Function>(f));
}


} // end detail
} // end agency

