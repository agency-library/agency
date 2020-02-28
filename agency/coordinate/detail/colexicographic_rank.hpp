#pragma once

#include <agency/detail/config.hpp>
#include <agency/coordinate/detail/index_cast.hpp>

namespace agency
{
namespace detail
{


// this function returns the rank of the given index in the index
// space implied by the shape argument
// its rank is its position in a colexicographically sorted array of all the indices in the index space
template<class Index, class Shape>
__AGENCY_ANNOTATION
std::size_t colexicographic_rank(const Index& idx, const Shape& shape)
{
  return agency::detail::index_cast<std::size_t>(idx, shape, agency::detail::index_space_size(shape));
}


} // end detail
} // end agency

