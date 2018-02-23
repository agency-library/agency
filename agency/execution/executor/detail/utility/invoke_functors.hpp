#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/unit.hpp>

namespace agency
{
namespace detail
{


// this functor is used by bulk_*_execute_with_void_result()
// this definition is used when there is a non-void predecessor parameter
template<class Function, class Predecessor = void>
struct ignore_unit_result_parameter_and_invoke
{
  mutable Function f;

  __agency_exec_check_disable__
  ignore_unit_result_parameter_and_invoke() = default;

  __agency_exec_check_disable__
  ignore_unit_result_parameter_and_invoke(const ignore_unit_result_parameter_and_invoke&) = default;

  __agency_exec_check_disable__
  ~ignore_unit_result_parameter_and_invoke() = default;

  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }
};

// this is the specialization used when there is no predecessor parameter
template<class Function>
struct ignore_unit_result_parameter_and_invoke<Function,void>
{
  mutable Function f;

  __agency_exec_check_disable__
  ignore_unit_result_parameter_and_invoke() = default;

  __agency_exec_check_disable__
  ignore_unit_result_parameter_and_invoke(const ignore_unit_result_parameter_and_invoke&) = default;

  __agency_exec_check_disable__
  ~ignore_unit_result_parameter_and_invoke() = default;

  template<class Index, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, unit&, SharedParameters&... shared_parameters) const
  {
    agency::detail::invoke(f, idx, shared_parameters...);
  }
};


// this functor is used by bulk_*_execute_with_collected_result()
// this definition is used when there is a non-void predecessor parameter
template<class Function, class Predecessor = void>
struct invoke_and_collect_result
{
  mutable Function f;

  __agency_exec_check_disable__
  template<class Index, class Collection, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Predecessor& predecessor, Collection& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, predecessor, shared_parameters...);
  }
};

// this is the specialization used when there is no predecessor parameter
template<class Function>
struct invoke_and_collect_result<Function,void>
{
  mutable Function f;

  __agency_exec_check_disable__
  template<class Index, class Collection, class... SharedParameters>
  __AGENCY_ANNOTATION
  void operator()(const Index& idx, Collection& results, SharedParameters&... shared_parameters) const
  {
    results[idx] = agency::detail::invoke(f, idx, shared_parameters...);
  }
};


} // end detail
} // end agency

