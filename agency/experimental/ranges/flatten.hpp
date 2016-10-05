#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/view.hpp> // for all()
#include <type_traits>
#include <utility>


namespace agency
{
namespace experimental
{


// flatten_view does not assume the size of the segments are the same
// operator[] and size() would have more efficient implementations if
// we made that assumption
// we should consider another kind of fancy range that would "un-tile" a collection of tiles
template<class Range>
class flatten_view
{
  private:
    using all_t = decltype(
      agency::experimental::all(std::declval<Range&>())
    );

  public:
    using difference_type = range_difference_t<all_t>;
    using value_type = range_value_t<all_t>;
    using reference = value_type&;

    __AGENCY_ANNOTATION
    flatten_view(const flatten_view&) = default;

    template<class RangeOfRanges,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 // XXX for some reason, range_value_t isn't SFINAE friendly, so workaround it
                 //range_value_t<RangeOfRanges>,
                 decltype(*std::declval<RangeOfRanges>().begin()),
                 all_t
               >::value
             )
            >
    __AGENCY_ANNOTATION
    flatten_view(RangeOfRanges&& ranges)
    {
      segments_.reserve(ranges.size());

      for(auto& rng : ranges)
      {
        segments_.push_back(all(rng));
      }
    }

  private:
    __AGENCY_ANNOTATION
    reference bracket_operator(size_t element_idx, size_t current_segment_idx) const
    {
      auto& segment = segments_[current_segment_idx];
      auto size = segment.size();

      // if the element is within the current segment, return it
      // otherwise, recurse
      // note that attempting to index an element that lies beyond the end of this view
      // will not terminate the recursion
      return element_idx < size ?
        segment[element_idx] :
        bracket_operator(element_idx - size, current_segment_idx + 1);
    }

  public:
    __AGENCY_ANNOTATION
    reference operator[](size_t i) const
    {
      // seems like we have to do a linear search through the segments
      // so, it't not clear this can be computed in O(1)
      // OTOH, it's not O(N) either (N being the total number of elements viewed by this view)
      return bracket_operator(i, 0);
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      size_t result = 0;
      for(auto& segment : segments_)
      {
        result += segment.size();
      }

      return result;
    }

  private:
    std::vector<all_t> segments_;
};


template<class RangeOfRanges>
__AGENCY_ANNOTATION
flatten_view<range_value_t<RangeOfRanges>> flatten(RangeOfRanges&& ranges)
{
  return flatten_view<range_value_t<RangeOfRanges>>(std::forward<RangeOfRanges>(ranges));
}


} // end experimental
} // end agency

