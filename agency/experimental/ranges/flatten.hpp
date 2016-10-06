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

    template<class> friend class flatten_view;

  public:
    using difference_type = range_difference_t<all_t>;
    using size_type = range_size_t<all_t>;
    using value_type = range_value_t<all_t>;
    using reference = range_reference_t<all_t>;

    flatten_view() = default;

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
    flatten_view(RangeOfRanges&& ranges)
    {
      segments_.reserve(ranges.size());

      for(auto& rng : ranges)
      {
        segments_.push_back(all(rng));
      }
    }

    // converting copy constructor
    template<class OtherRange,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 all_t,
                 typename flatten_view<OtherRange>::all_t
               >::value
             )>
    flatten_view(const flatten_view<OtherRange>& other)
      : segments_(other.segments_.begin(), other.segments_.end())
    {}

  private:
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
    reference operator[](size_type i) const
    {
      // seems like we have to do a linear search through the segments
      // so, it't not clear this can be computed in O(1)
      // OTOH, it's not O(N) either (N being the total number of elements viewed by this view)
      return bracket_operator(i, 0);
    }

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
        reference operator*() const
        {
          return self_[current_position_];
        }

        // pre-increment
        iterator operator++()
        {
          ++current_position_;
          return *this;
        }

        // pre-decrement
        iterator operator--()
        {
          --current_position_;
          return *this;
        }

        // post-increment
        iterator operator++(int)
        {
          iterator result = *this;
          current_position_++;
          return result;
        }

        // post-decrement
        iterator operator--(int)
        {
          iterator result = *this;
          current_position_--;
          return result;
        }

        // add-assign
        iterator operator+=(size_type n)
        {
          current_position_ += n;
          return *this;
        }

        // minus-assign
        iterator operator-=(size_type n)
        {
          current_position_ -= n;
          return *this;
        }

        // add
        iterator operator+(size_type n)
        {
          iterator result = *this;
          result += n;
          return result;
        }

        // minus
        iterator operator-(size_type n)
        {
          iterator result = *this;
          result -= n;
          return result;
        }

        // bracket
        reference operator[](size_type n)
        {
          iterator tmp = *this + n;
          return *tmp;
        }

        // equal
        bool operator==(const iterator& rhs) const
        {
          // we assume that *this and rhs came from the same flattened_view,
          // so we do not compare their self_ members
          return current_position_ == rhs.current_position_;
        }

        // not equal
        bool operator!=(const iterator& rhs) const
        {
          return !(*this == rhs);
        }

        // difference
        difference_type operator-(const iterator& rhs) const
        {
          return current_position_ - rhs.current_position_;
        }

      private:
        friend flatten_view;

        iterator(size_type current_position, const flatten_view& self)
          : current_position_(current_position),
            self_(self)
        {}

        // XXX a more efficient implementation would track the current segment
        // XXX and the current position within the segment
        //     could keep an iterator to the current segment
        //     would make operator- and operator+= less efficient because they would involve linear searches
        size_type current_position_;

        // it's expensive to keep a copy of the flatten_view from whence
        // this iterator came, but it'too common for the original flatten_view's
        // lifetime to end while this iterator is still alive
        const flatten_view self_;
    };

    iterator begin() const
    {
      return iterator(0, *this);
    }

    iterator end() const
    {
      return iterator(size(), *this);
    }

  private:
    // XXX we may wish to use agency::detail::array so that we can use __AGENCY_ANNOTATION
    std::vector<all_t> segments_;
};


// XXX I think this should actually return something like
// flatten_view<range_reference_t<RangeOfRanges>
template<class RangeOfRanges>
flatten_view<range_value_t<RangeOfRanges>> flatten(RangeOfRanges&& ranges)
{
  return flatten_view<range_value_t<RangeOfRanges>>(std::forward<RangeOfRanges>(ranges));
}


} // end experimental
} // end agency

