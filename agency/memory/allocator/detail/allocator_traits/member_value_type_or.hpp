#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


// returns T::value_type if it exists, Default otherwise
template<class T, class Default>
struct member_value_type_or
{
  template<class U>
  using helper = typename U::value_type;

  using type = detected_or_t<Default, helper, T>;
};

template<class T, class Default>
using member_value_type_or_t = typename member_value_type_or<T,Default>::type;


} // end detail
} // end agency

