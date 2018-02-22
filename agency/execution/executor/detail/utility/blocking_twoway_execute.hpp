#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/detail/adaptors/executor_ref.hpp>
#include <agency/execution/executor/properties/always_blocking.hpp>
#include <agency/execution/executor/properties/single.hpp>
#include <agency/execution/executor/properties/twoway.hpp>
#include <agency/execution/executor/require.hpp>


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
  // get a reference to avoid copies in agency::require
  executor_ref<E> exec_ref{exec};

  return agency::require(exec_ref, always_blocking, single, twoway).twoway_execute(std::forward<Function>(f)).get();
}


} // end detail
} // end agency

