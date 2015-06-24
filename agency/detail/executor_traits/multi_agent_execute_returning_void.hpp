#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/discarding_container.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Executor, class Function>
void multi_agent_execute_returning_void(std::true_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  return ex.execute(f, shape);
} // end multi_agent_execute_returning_void()


template<class Executor, class Function>
void multi_agent_execute_returning_void(std::false_type, Executor& ex, Function f, typename new_executor_traits<Executor>::shape_type shape)
{
  auto g = [=](const typename new_executor_traits<Executor>::index_type& idx)
  {
    f(idx);

    // return something which can be cheaply discarded
    return 0;
  };

  new_executor_traits<Executor>::template execute<discarding_container>(ex, g, shape);
} // end multi_agent_execute_returning_void()


} // end new_executor_traits_detail
} // end detail


template<class Executor>
  template<class Function,
           class Enable>
void new_executor_traits<Executor>
  ::execute(typename new_executor_traits<Executor>::executor_type& ex,
            Function f,
            typename new_executor_traits<Executor>::shape_type shape)
{
  using check_for_member_function = detail::new_executor_traits_detail::has_multi_agent_execute_returning_void<
    Executor,
    Function
  >;

  return detail::new_executor_traits_detail::multi_agent_execute_returning_void(check_for_member_function(), ex, f, shape);
} // end new_executor_traits::execute()


} // end agency

