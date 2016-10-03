#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>

namespace agency
{
namespace detail
{


template<class T, class U, template<class> class Default>
struct member_future_or
{
  template<class V>
  using helper = typename V::template future<U>;

  using type = detected_or_t<Default<U>, helper, T>;
};

template<class T, class U, template<class> class Default>
using member_future_or_t = typename member_future_or<T,U,Default>::type;


} // end detail
} // end agency

