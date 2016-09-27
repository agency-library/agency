#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/new_executor_traits/executor_shape.hpp>
#include <agency/execution/executor/detail/executor_traits/container_factory.hpp>
#include <agency/execution/executor/detail/utility/executor_container_or_void.hpp>
#include <type_traits>

namespace agency
{
namespace detail
{


// this type is just a placeholder type which indicates special cases
// of bulk_invoke() or bulk_async() where no container is required because
// the user function returns void
struct void_factory {};

template<class ResultOfFunction, class Executor,
         class = typename std::enable_if<
           std::is_void<ResultOfFunction>::value
         >::type>
__AGENCY_ANNOTATION
void_factory make_result_factory(const Executor&)
{
  return void_factory{};
}


template<class ResultOfFunction, class Executor,
         class = typename std::enable_if<
           std::is_void<ResultOfFunction>::value
         >::type>
__AGENCY_ANNOTATION
void_factory new_make_result_factory(const Executor&, const executor_shape_t<Executor>&)
{
  return void_factory{};
}


template<class Executor, class ResultOfFunction>
struct result_container
{
  using type = typename std::conditional<
    is_scope_result<ResultOfFunction>::value,
    typename scope_result_to_scope_result_container<ResultOfFunction, Executor>::type,
    new_executor_container_t<Executor,ResultOfFunction>
  >::type;
};

template<class Executor, class ResultOfFunction>
using result_container_t = typename result_container<Executor, ResultOfFunction>::type;


template<class ResultOfFunction, class Executor,
         class = typename std::enable_if<
           !std::is_void<ResultOfFunction>::value
         >::type>
__AGENCY_ANNOTATION
detail::executor_traits_detail::container_factory<result_container_t<Executor,ResultOfFunction>>
  make_result_factory(const Executor&)
{
  // compute the type of container to use to store results
  using container_type = result_container_t<Executor,ResultOfFunction>;

  // create a factory for the result container
  return detail::executor_traits_detail::container_factory<container_type>();
}


template<class ResultOfFunction, class Executor,
         class = typename std::enable_if<
           !std::is_void<ResultOfFunction>::value
         >::type>
__AGENCY_ANNOTATION
construct<result_container_t<Executor,ResultOfFunction>, executor_shape_t<Executor>>
  new_make_result_factory(const Executor&, const executor_shape_t<Executor>& shape)
{
  // compute the type of container to use to store results
  using container_type = result_container_t<Executor,ResultOfFunction>;

  // create a factory for the result container that calls the container's constructor with the given shape
  return make_construct<container_type>(shape);
}


} // end detail
} // end agency

