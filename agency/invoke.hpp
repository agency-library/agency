#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/control_structures/bind.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/utility/blocking_twoway_execute.hpp>
#include <agency/execution/executor/sequenced_executor.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>

namespace agency
{


template<class Executor, class Function, class... Args>
__AGENCY_ANNOTATION
detail::result_of_t<
  typename std::decay<Function&&>::type(typename std::decay<Args&&>::type...)
>
  invoke(Executor& exec, Function&& f, Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), std::forward<Args>(args)...);

  return detail::blocking_twoway_execute(exec, std::move(g));
}


template<class Function, class... Args>
detail::result_of_t<
  typename std::decay<Function&&>::type(typename std::decay<Args&&>::type...)
>
  invoke(Function&& f, Args&&... args)
{
  agency::sequenced_executor exec;
  return agency::invoke(exec, std::forward<Function>(f), std::forward<Args>(args)...);
}


} // end agency

