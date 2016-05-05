#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
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


template<class Executor, class ResultOfFunction>
struct result_container
{
  using type = typename std::conditional<
    detail::is_scope_result<ResultOfFunction>::value,
    typename detail::scope_result_to_scope_result_container<ResultOfFunction, Executor>::type,
    executor_result_t<Executor,ResultOfFunction>
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


} // end detail
} // end agency

