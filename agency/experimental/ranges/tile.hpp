#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/chunk.hpp>

namespace agency
{
namespace experimental
{



// tile() is a synonym for chunk()
// introduce this while we bikeshed the names
template<class Range, class Difference>
__AGENCY_ANNOTATION
auto tile(Range&& rng, Difference tile_size) ->
  decltype(
    agency::experimental::chunk(std::forward<Range>(rng), tile_size)
  )
{
  return agency::experimental::chunk(std::forward<Range>(rng), tile_size);
}


// tile_evenly() is a synonym for chunk_evenly()
// introduce this while we bikeshed the names
template<class Range, class Difference>
__AGENCY_ANNOTATION
auto tile_evenly(Range&& rng, Difference desired_number_of_tiles) ->
  decltype(
    agency::experimental::tile(std::forward<Range>(rng), std::declval<Difference>())
  )
{
  // note that this calculation will not necessarily result in the desired number of tiles
  // in general, there is no way to partition a range into exactly N-1 equally-sized tiles plus a single odd-sized tile
  Difference tile_size = (rng.size() + desired_number_of_tiles - 1) / desired_number_of_tiles;
  return agency::experimental::tile(std::forward<Range>(rng), tile_size);
}


} // end experimental
} // end agency

