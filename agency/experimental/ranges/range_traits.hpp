#pragma once

#include <cstddef>
#include <limits>
#include <iterator>
#include <utility>
#include <agency/detail/config.hpp>

namespace agency
{
namespace experimental
{


template<class Range>
struct range_iterator
{
  using type = decltype(std::declval<Range&>().begin());
};

template<class Range>
using range_iterator_t = typename range_iterator<Range>::type;


template<class Range>
struct range_sentinel
{
  using type = decltype(std::declval<Range&>().end());
};

template<class Range>
using range_sentinel_t = typename range_sentinel<Range>::type;


template<class Range>
struct range_value
{
  using type = typename std::iterator_traits<range_iterator_t<Range>>::value_type;
};

template<class Range>
using range_value_t = typename range_value<Range>::type;


template<class Range>
struct range_reference
{
  using type = typename std::iterator_traits<range_iterator_t<Range>>::reference;
};

template<class Range>
using range_reference_t = typename range_reference<Range>::type;


template<class Range>
struct range_difference
{
  using type = typename std::iterator_traits<range_iterator_t<Range>>::difference_type;
};

template<class Range>
using range_difference_t = typename range_difference<Range>::type;


template<class Range>
struct range_size
{
  // XXX range-v3 uses this definition:
  // using type = make_unsigned<range_difference_t<Range>>;
  using type = decltype(std::declval<Range&>().size());
};

template<class Range>
using range_size_t = typename range_size<Range>::type;


enum cardinality
{
  infinite = -3,
  unknown = -2,
  finite = -1,
  _max_ = std::numeric_limits<int>::max()
};


// by default, assume all ranges are finite
template<class Range>
struct range_cardinality : std::integral_constant<cardinality, finite> {};


// make range_cardinality automatically decay
template<class Range>
struct range_cardinality<const Range>
  : range_cardinality<Range>
{};

template<class Range>
struct range_cardinality<Range&>
  : range_cardinality<Range>
{};


template<class Range>
using is_statically_sized_range = std::integral_constant<bool, (range_cardinality<Range>::value > finite)>;


} // end experimental
} // end agency

