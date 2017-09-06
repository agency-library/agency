#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/type_list.hpp>
#include <agency/detail/scoped_in_place_type.hpp>
#include <agency/execution/executor/executor_traits/detail/member_barrier_type_or.hpp>
#include <agency/execution/executor/executor_traits/executor_execution_depth.hpp>


namespace agency
{
namespace detail
{


template<class Executor>
struct executor_barrier_types_as_scoped_in_place_type_t_impl
{
  // ask the executor what its barrier is, if it has one
  using barrier_type_or_void = member_barrier_type_or_t<Executor,void>;

  // turn this into an instance of scoped_in_place_type_t
  using scoped_barrier_type = make_scoped_in_place_type_t<barrier_type_or_void>;

  // get the list of barriers or void as a type_list
  using list_of_barriers_or_void = agency::detail::type_list_of_template_parameters<scoped_barrier_type>;

  // extend the list with void such that its length matches the Executor's execution_depth
  static const std::size_t result_depth = agency::executor_execution_depth<Executor>::value;
  static const std::size_t difference = result_depth - agency::detail::type_list_size<list_of_barriers_or_void>::value;
  using void_types = agency::detail::type_list_repeat<difference, void>;

  // return the result as a scoped_in_place_type_t
  using type = agency::detail::type_list_instantiate<scoped_in_place_type_t, agency::detail::type_list_cat<list_of_barriers_or_void, void_types>>;


  // ensure we computed a sensible result
  using test_type_list = agency::detail::type_list_of_template_parameters<type>;
  static_assert(result_depth == agency::detail::type_list_size<test_type_list>::value, "make_scoped_in_place_barrier_t<Executor>'s depth must match executor_execution_depth<Executor>.");
};

template<class Executor>
using executor_barrier_types_as_scoped_in_place_type_t = typename executor_barrier_types_as_scoped_in_place_type_t_impl<Executor>::type;


} // end detail
} // end agency

