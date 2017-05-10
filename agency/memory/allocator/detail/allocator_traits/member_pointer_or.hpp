#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


// returns T::pointer if it exists, Default otherwise
template<class T, class Default>
struct member_pointer_or
{
  template<class U>
  using helper = typename U::pointer;

  using type = detected_or_t<Default, helper, T>;
};

template<class T, class Default>
using member_pointer_or_t = typename member_pointer_or<T,Default>::type;


} // end detail
} // end agency



