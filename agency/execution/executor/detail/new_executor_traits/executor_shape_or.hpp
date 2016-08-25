#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <cstddef>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class Default = size_t>
struct executor_shape_or
{
  template<class U>
  using helper = typename U::shape_type;

  using type = detected_or_t<Default, helper, T>;
};

template<class T, class Default = size_t>
using executor_shape_or_t = typename executor_shape_or<T,Default>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency

