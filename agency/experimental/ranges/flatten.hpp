#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/all.hpp>
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
template<class RangeOfRanges>
class flatten_view
{
  private:
    using inner_range_type = range_reference_t<RangeOfRanges>;
    using segments_t = agency::experimental::all_t<RangeOfRanges&>;

    template<class> friend class flatten_view;

  public:
    using difference_type = range_difference_t<inner_range_type>;
    using size_type = range_size_t<inner_range_type>;
    using value_type = range_value_t<inner_range_type>;
    using reference = range_reference_t<inner_range_type>;

    flatten_view() = default;
    flatten_view(const flatten_view&) = default;

    template<class OtherRangeOfRanges,
             __AGENCY_REQUIRES(
               !std::is_same<
                 typename std::decay<OtherRangeOfRanges>::type,
                 flatten_view
               >::value
             ),
             __AGENCY_REQUIRES(
               std::is_convertible<
                 experimental::all_t<OtherRangeOfRanges>,
                 segments_t
               >::value
             )
            >
    __AGENCY_ANNOTATION
    flatten_view(OtherRangeOfRanges&& ranges)
      : segments_(agency::experimental::all(std::forward<OtherRangeOfRanges>(ranges)))
    {}

    // converting copy constructor
    template<class OtherRangeOfRanges,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 segments_t,
                 typename flatten_view<OtherRangeOfRanges>::segments_t
               >::value
             )>
    __AGENCY_ANNOTATION
    flatten_view(const flatten_view<OtherRangeOfRanges>& other)
      : segments_(other.segments_)
    {}

  private:
    __AGENCY_ANNOTATION
    reference bracket_operator(size_type element_idx, size_t current_segment_idx) const
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
    reference operator[](size_type i) const
    {
      // seems like we have to do a linear search through the segments
      // so, it't not clear this can be computed in O(1)
      // OTOH, it's not O(N) either (N being the total number of elements viewed by this view)
      return bracket_operator(i, 0);
    }

    __AGENCY_ANNOTATION
    size_type size() const
    {
      size_type result = 0;
      for(auto& segment : segments_)
      {
        result += segment.size();
      }

      return result;
    }

    class iterator
    {
      public:
        using value_type = typename flatten_view::value_type;
        using reference = typename flatten_view::reference;
        using difference_type = typename flatten_view::difference_type;
        using pointer = value_type*;
        using iterator_category = std::random_access_iterator_tag;

        // dereference
        __AGENCY_ANNOTATION
        reference operator*() const
        {
          return self_[current_position_];
        }

        // pre-increment
        __AGENCY_ANNOTATION
        iterator operator++()
        {
          ++current_position_;
          return *this;
        }

        // pre-decrement
        __AGENCY_ANNOTATION
        iterator operator--()
        {
          --current_position_;
          return *this;
        }

        // post-increment
        __AGENCY_ANNOTATION
        iterator operator++(int)
        {
          iterator result = *this;
          current_position_++;
          return result;
        }

        // post-decrement
        __AGENCY_ANNOTATION
        iterator operator--(int)
        {
          iterator result = *this;
          current_position_--;
          return result;
        }

        // add-assign
        __AGENCY_ANNOTATION
        iterator operator+=(size_type n)
        {
          current_position_ += n;
          return *this;
        }

        // minus-assign
        __AGENCY_ANNOTATION
        iterator operator-=(size_type n)
        {
          current_position_ -= n;
          return *this;
        }

        // add
        __AGENCY_ANNOTATION
        iterator operator+(size_type n)
        {
          iterator result = *this;
          result += n;
          return result;
        }

        // minus
        __AGENCY_ANNOTATION
        iterator operator-(size_type n)
        {
          iterator result = *this;
          result -= n;
          return result;
        }

        // bracket
        __AGENCY_ANNOTATION
        reference operator[](size_type n)
        {
          iterator tmp = *this + n;
          return *tmp;
        }

        // equal
        __AGENCY_ANNOTATION
        bool operator==(const iterator& rhs) const
        {
          // we assume that *this and rhs came from the same flattened_view,
          // so we do not compare their self_ members
          return current_position_ == rhs.current_position_;
        }

        // not equal
        __AGENCY_ANNOTATION
        bool operator!=(const iterator& rhs) const
        {
          return !(*this == rhs);
        }

        // difference
        __AGENCY_ANNOTATION
        difference_type operator-(const iterator& rhs) const
        {
          return current_position_ - rhs.current_position_;
        }

      private:
        friend flatten_view;

        __AGENCY_ANNOTATION
        iterator(size_type current_position, const flatten_view& self)
          : current_position_(current_position),
            self_(self)
        {}

        // XXX a more efficient implementation would track the current segment
        // XXX and the current position within the segment
        //     could keep an iterator to the current segment
        //     would make operator- and operator+= less efficient because they would involve linear searches
        size_type current_position_;

        flatten_view self_;
    };

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return iterator(0, *this);
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return iterator(size(), *this);
    }

    __AGENCY_ANNOTATION
    flatten_view all() const
    {
      return *this;
    }

  private:
    segments_t segments_;

  public:
    __AGENCY_ANNOTATION
    auto segment(size_type i) const
      -> decltype(agency::experimental::all(this->segments_[i]))
    {
      return agency::experimental::all(segments_[i]);
    }

    __AGENCY_ANNOTATION
    const segments_t& segments() const
    {
      return segments_;
    }
};


template<class RangeOfRanges>
__AGENCY_ANNOTATION
flatten_view<RangeOfRanges> flatten(RangeOfRanges&& ranges)
{
  return flatten_view<RangeOfRanges>(std::forward<RangeOfRanges>(ranges));
}


} // end experimental
} // end agency

