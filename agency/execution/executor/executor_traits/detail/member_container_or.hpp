#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


template<class T, class U, template<class> class Default>
struct member_container_or
{
  template<class V>
  using helper = typename V::template container<U>;

  using type = detected_or_t<Default<U>, helper, T>;
};

template<class T, class U, template<class> class Default>
using member_container_or_t = typename member_container_or<T,U,Default>::type;


} // end detail
} // end agency

