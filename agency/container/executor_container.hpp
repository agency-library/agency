#pragma once

#include <agency/detail/config.hpp>
#include <agency/execution/executor/executor_traits/executor_index.hpp>
#include <agency/execution/executor/executor_traits/executor_shape.hpp>
#include <agency/execution/executor/executor_traits/executor_allocator.hpp>
#include <agency/experimental/ndarray.hpp>

namespace agency
{


// XXX eliminate this alias in favor of a type e.g. bulk_result
template<class Executor, class T>
struct executor_container
{
  using type = experimental::basic_ndarray<T, executor_shape_t<Executor>, executor_allocator_t<Executor,T>, executor_index_t<Executor>>;
};

template<class Executor, class T>
using executor_container_t = typename executor_container<Executor,T>::type;


} // end agency

