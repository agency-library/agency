#pragma once

#include <agency/coordinate.hpp>
#include <cstddef>


namespace agency
{
namespace detail
{


template<size_t rank>
struct default_shape
{
  using type = point<std::size_t,rank>;
}; // end default_shape

// when rank is 1, just return a simple size_t rather than a point<size_t,1>
template<>
struct default_shape<1>
{
  using type = std::size_t;
}; // end default_shape


// a rank 0 makes no sense, so don't define a result
template<>
struct default_shape<0>
{
}; // end default_shape

template<size_t rank>
using default_shape_t = typename default_shape<rank>::type;


} // end detail
} // end agency

