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
auto tile_evenly(Range&& rng, Difference number_of_chunks) ->
  decltype(
    agency::experimental::tile(std::forward<Range>(rng), std::declval<Difference>())
  )
{
  Difference chunk_size = (rng.size() + number_of_chunks - 1) / number_of_chunks;
  return agency::experimental::tile(std::forward<Range>(rng), chunk_size);
}


} // end experimental
} // end agency

