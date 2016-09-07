#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{
namespace new_executor_traits_detail
{


template<class T, class Default>
struct member_execution_category_or
{
  template<class U>
  using helper = typename U::execution_category;

  using type = detected_or_t<Default, helper, T>;
};

template<class T, class Default>
using member_execution_category_or_t = typename member_execution_category_or<T,Default>::type;


} // end new_executor_traits_detail
} // end detail
} // end agency

