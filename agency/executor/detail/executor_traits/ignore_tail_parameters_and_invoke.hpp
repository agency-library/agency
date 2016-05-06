#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/executor/executor_traits.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Function>
struct ignore_tail_parameters_and_invoke
{
  mutable Function f;

  template<class Index, class... Args>
  __AGENCY_ANNOTATION
  result_of_t<Function(Index)>
  operator()(const Index& idx, Args&&...) const
  {
    return agency::detail::invoke(f, idx);
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
homogeneous_tuple<detail::unit_factory,executor_traits<Executor>::execution_depth> make_tuple_of_unit_factories(Executor&)
{
  return make_homogeneous_tuple<executor_traits<Executor>::execution_depth>(detail::unit_factory());
}


} // end executor_traits_detail
} // end detail
} // end agency

