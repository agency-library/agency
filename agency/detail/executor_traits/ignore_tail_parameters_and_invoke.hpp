#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/new_executor_traits.hpp>
#include <agency/detail/factory.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class Function>
struct ignore_tail_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  typename std::result_of<Function(Index)>::type
  operator()(const Index& idx, Args&&...) const
  {
    // XXX should use std::invoke
    return f(idx);
  }
};


template<class Function>
__AGENCY_ANNOTATION
ignore_tail_parameters_and_invoke<Function> make_ignore_tail_parameters_and_invoke(Function f)
{
  return ignore_tail_parameters_and_invoke<Function>{f};
} // end make_ignore_tail_parameters_and_invoke()


template<class Executor>
__AGENCY_ANNOTATION
homogeneous_tuple<detail::unit_factory,new_executor_traits<Executor>::execution_depth> make_tuple_of_unit_factories(Executor&)
{
  return make_homogeneous_tuple<new_executor_traits<Executor>::execution_depth>(detail::unit_factory());
}


} // end new_executor_traits_detail
} // end detail
} // end agency

