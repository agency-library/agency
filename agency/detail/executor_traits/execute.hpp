#pragma once

#include <agency/detail/config.hpp>
#include <agency/new_executor_traits.hpp>
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


template<class Executor, class Function>
struct has_single_agent_execute_impl
{
  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().execute(
               std::declval<Function>()
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(int);

  using type = decltype(test<Executor>(0));
};

template<class Executor, class Function>
using has_single_agent_execute = typename has_single_agent_execute_impl<Executor,Function>::type;


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

