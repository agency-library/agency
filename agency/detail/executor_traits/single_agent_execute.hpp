#pragma once

#include <agency/detail/config.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/check_for_member_functions.hpp>
#include <agency/detail/executor_traits/single_element_container.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <agency/detail/shape_cast.hpp>
#include <agency/functional.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Function>
typename std::result_of<Function()>::type
  single_agent_execute(std::true_type, Executor& ex, Function&& f)
{
  return ex.execute(std::forward<Function>(f));
} // end single_agent_execute()


template<class Executor, class Function>
__AGENCY_ANNOTATION
typename std::result_of<Function()>::type
  single_agent_execute_impl(Executor& ex, Function&& f,
                            typename std::enable_if<
                              std::is_void<
                                typename std::result_of<Function()>::type
                              >::value
                            >::type* = 0)
{
  using shape_type = typename executor_traits<Executor>::shape_type;
  using index_type = typename executor_traits<Executor>::index_type;

  executor_traits<Executor>::execute(ex, [&](const index_type&)
  {
    agency::invoke(std::forward<Function>(f));
  },
  detail::shape_cast<shape_type>(1));
}


template<class Executor, class Function>
__AGENCY_ANNOTATION
typename std::result_of<Function()>::type
  single_agent_execute_impl(Executor& ex, Function&& f,
                            typename std::enable_if<
                              !std::is_void<
                                typename std::result_of<Function()>::type
                              >::value
                            >::type* = 0)
{
  using value_type = typename std::result_of<Function()>::type;
  using shape_type = typename executor_traits<Executor>::shape_type;

  using container_type = single_element_container<value_type,shape_type>;

  using index_type = typename executor_traits<Executor>::index_type;

  return executor_traits<Executor>::execute(ex, [&](const index_type&)
  {
    return agency::invoke(std::forward<Function>(f));
  },
  container_factory<container_type>{},
  detail::shape_cast<shape_type>(1)).element;
}


template<class Executor, class Function>
__AGENCY_ANNOTATION
typename std::result_of<Function()>::type
  single_agent_execute(std::false_type, Executor& ex, Function&& f)
{
  return executor_traits_detail::single_agent_execute_impl(ex, std::forward<Function>(f));
} // end single_agent_execute()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Function>
__AGENCY_ANNOTATION
typename std::result_of<Function()>::type
  executor_traits<Executor>
    ::execute(typename executor_traits<Executor>::executor_type& ex,
              Function&& f)
{
  using check_for_member_function = detail::executor_traits_detail::has_single_agent_execute<
    Executor,
    Function
  >;

  return detail::executor_traits_detail::single_agent_execute(check_for_member_function(), ex, std::forward<Function>(f));
} // end executor_traits::execute()


} // end agency

