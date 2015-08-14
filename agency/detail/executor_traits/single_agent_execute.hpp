#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/single_element_container.hpp>
#include <agency/detail/shape_cast.hpp>
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
  single_agent_execute_impl(Executor& ex, Function f,
                            typename std::enable_if<
                              std::is_void<
                                typename std::result_of<Function()>::type
                              >::value
                            >::type* = 0)
{
  using shape_type = typename new_executor_traits<Executor>::shape_type;
  using index_type = typename new_executor_traits<Executor>::index_type;

  new_executor_traits<Executor>::execute(ex, [=](const index_type&)
  {
    // XXX should use std::invoke()
    f();
  },
  detail::shape_cast<shape_type>(1));
}


template<class Executor, class Function>
typename std::result_of<Function()>::type
  single_agent_execute_impl(Executor& ex, Function f,
                            typename std::enable_if<
                              !std::is_void<
                                typename std::result_of<Function()>::type
                              >::value
                            >::type* = 0)
{
  using value_type = typename std::result_of<Function()>::type;
  using container_type = single_element_container<value_type>;

  using shape_type = typename new_executor_traits<Executor>::shape_type;
  using index_type = typename new_executor_traits<Executor>::index_type;

  return new_executor_traits<Executor>::template execute<container_type>(ex, [=](const index_type&)
  {
    // XXX should use std::invoke()
    return f();
  },
  detail::shape_cast<shape_type>(1)).element;
}


template<class Executor, class Function>
typename std::result_of<Function()>::type
  single_agent_execute(std::false_type, Executor& ex, Function f)
{
  return new_executor_traits_detail::single_agent_execute_impl(ex, f);
} // end single_agent_execute()


} // end new_executor_traits_detail
} // end detail


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

