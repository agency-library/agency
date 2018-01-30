#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits.hpp>
#include <agency/execution/executor/detail/adaptors/bulk_twoway_executor.hpp>

namespace agency
{
namespace detail
{


template<class E, class Function, class ResultFactory, class... Factories,
         __AGENCY_REQUIRES(is_executor<E>::value),
         __AGENCY_REQUIRES(executor_execution_depth<E>::value == sizeof...(Factories))
        >
__AGENCY_ANNOTATION
executor_future_t<
  E,
  detail::result_of_t<ResultFactory()>
>
bulk_twoway_execute(const E& exec, Function f, executor_shape_t<E> shape, ResultFactory result_factory, Factories... shared_factories)
{
  return detail::bulk_twoway_executor<E>(exec).bulk_twoway_execute(f, shape, result_factory, shared_factories...);
}


} // end detail
} // end agency

