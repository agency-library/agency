#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/execution_functions/twoway_execute.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>


namespace agency
{
namespace detail
{


__agency_exec_check_disable__
template<class E, class Function, __AGENCY_REQUIRES(is_executor<E>::value)>
__AGENCY_ANNOTATION
detail::result_of_t<detail::decay_t<Function>()>
  blocking_twoway_execute(const E& exec, Function&& f)
{
  detail::always_blocking_executor<E> blocking_exec(exec);

  // XXX nomerge
  //     use agency::require()
  return detail::twoway_execute(blocking_exec, std::forward<Function>(f)).get();
}


} // end detail
} // end agency

