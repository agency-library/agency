#pragma once

#include <cstddef>
#include <limits>
#include <agency/detail/config.hpp>

namespace agency
{
namespace experimental
{


template<class Range>
struct range_size
{
  // XXX range-v3 uses this definition:
  // using type = make_unsigned<range_difference_t<Range>>;
  using type = decltype(std::declval<Range&>().size());
};

template<class Range>
using range_size_t = typename range_size<Range>::type;


} // end experimental
} // end agency

