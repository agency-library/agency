#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
typename std::result_of<Function()>::type
  single_agent_execute(std::true_type, Executor& ex, Function f)
{
  return ex.execute(f);
} // end single_agent_execute()


template<class Executor, class Function>
typename std::result_of<Function()>::type
  single_agent_execute(std::false_type, Executor& ex, Function f)
{
  auto fut = new_executor_traits<Executor>::async_execute(ex, f);

  // XXX should use an executor_traits operation on the future rather than .get()
  return fut.get();
} // end single_agent_execute()


} // end detail
} // end new_executor_traits_detail


template<class Executor>
  template<class Function>
typename std::result_of<Function()>::type
  new_executor_traits<Executor>
    ::execute(typename new_executor_traits<Executor>::executor_type& ex,
              Function f)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_single_agent_execute<
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::single_agent_execute(check_for_member_function(), ex, f);
} // end new_executor_traits::execute()


} // end agency

