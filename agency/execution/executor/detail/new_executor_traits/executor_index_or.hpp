#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/detail/new_executor_traits/executor_shape_or.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class Default = executor_shape_or_t<T>>
struct executor_index_or
{
  template<class U>
  using helper = typename U::index_type;

  using type = detected_or_t<Default, helper, T>;
};

template<class T, class Default = executor_shape_or_t<T>>
using executor_index_or_t = typename executor_index_or<T,Default>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency

