#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <agency/experimental/ranges/transformed.hpp>
#include <agency/experimental/short_vector.hpp>
#include <type_traits>
#include <utility>
#include <cassert>


namespace agency
{
namespace experimental
{


template<class RangeOfRanges, size_t max_tile_count_ = 8>
class small_untiled_view
{
  private:
    using inner_range_type = range_reference_t<RangeOfRanges>;

    // the type of types we store is the all() type of the inner range of the RangeOfRanges
    using tile_view_type = all_t<inner_range_type>;

    template<class,size_t> friend class small_untiled_view;

  public:
    // XXX this should probably be size_type (which could be narrow, or signed)
    //     rather than size_t
    static constexpr size_t max_tile_count = max_tile_count_;

    using difference_type = range_difference_t<tile_view_type>;
    using size_type = range_size_t<tile_view_type>;
    using value_type = range_value_t<tile_view_type>;
    using reference = range_reference_t<tile_view_type>;

    small_untiled_view() = default;

    small_untiled_view(const small_untiled_view&) = default;

    template<class OtherRangeOfRanges,
             __AGENCY_REQUIRES(
               std::is_convertible<
                 all_t<range_reference_t<OtherRangeOfRanges>>,
                 tile_view_type
               >::value
             )>
    __AGENCY_ANNOTATION
    small_untiled_view(const small_untiled_view<OtherRangeOfRanges>& other)
      : small_untiled_view(other.tile_size_, other.tiles())
    {}

    template<class OtherRangeOfRanges,
             __AGENCY_REQUIRES(
                std::is_convertible<
                  all_t<range_reference_t<OtherRangeOfRanges>>,
                  tile_view_type
                >::value
             )>
    __AGENCY_ANNOTATION
    small_untiled_view(size_t tile_size, OtherRangeOfRanges&& tiles)
      : tile_size_(tile_size),
        tiles_(transformed(agency::experimental::all, std::forward<OtherRangeOfRanges>(tiles)))
    {}

    __AGENCY_ANNOTATION
    reference bracket_operator(std::integral_constant<size_t,max_tile_count-1>, size_t i) const
    {
      return tiles_[max_tile_count-1][i];
    }

    template<size_t tile_idx>
    __AGENCY_ANNOTATION
    reference bracket_operator(std::integral_constant<size_t,tile_idx>, size_t i) const
    {
      return i < tile_size_ ? tiles_[tile_idx][i] : bracket_operator(std::integral_constant<size_t,tile_idx+1>(), i - tile_size_);
    }

    __AGENCY_ANNOTATION
    reference operator[](size_t i) const
    {
      // the performance of operator[] depends on the tiles_ array being statically indexed
      return bracket_operator(std::integral_constant<size_t,0>(), i);
    }

    __AGENCY_ANNOTATION
    size_t size() const
    {
      if(tiles_.empty()) return 0;

      return tile_size_ * (tiles_.size() - 1) + tiles_.back().size();
    }

    // this iterator type is trivial:
    // it just copies the view it came from (self_) and tracks its current position
    // XXX might want to refactor this into detail::view_iterator or something because
    //     it is repeated inside of flattened_view
    class iterator
    {
      public:
        using value_type = typename small_untiled_view::value_type;
        using reference = typename small_untiled_view::reference;
        using difference_type = typename small_untiled_view::difference_type;
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
        friend small_untiled_view;

        __AGENCY_ANNOTATION
        iterator(size_type current_position, const small_untiled_view& self)
          : current_position_(current_position),
            self_(self)
        {}

        // XXX a more efficient implementation would track the current tile
        // XXX and the current position within the tile
        //     could keep an iterator to the current tile
        //     would make operator- and operator+= less efficient because they would involve linear searches
        size_type current_position_;

        small_untiled_view self_;
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
    small_untiled_view all() const
    {
      return *this;
    }

  private:
    size_t tile_size_;
    mutable short_vector<tile_view_type,max_tile_count> tiles_;

  public:
    __AGENCY_ANNOTATION
    auto tiles() const ->
      decltype(agency::experimental::all(this->tiles_))
    {
      return agency::experimental::all(this->tiles_);
    }
};


template<size_t max_num_tiles, class RangeOfRanges>
__AGENCY_ANNOTATION
small_untiled_view<RangeOfRanges, max_num_tiles> untile(size_t tile_size, RangeOfRanges&& tiles)
{
  return small_untiled_view<RangeOfRanges,max_num_tiles>(tile_size, std::forward<RangeOfRanges>(tiles));
}


template<class RangeOfRanges>
__AGENCY_ANNOTATION
small_untiled_view<RangeOfRanges> untile(size_t tile_size, RangeOfRanges&& tiles)
{
  return small_untiled_view<RangeOfRanges>(tile_size, std::forward<RangeOfRanges>(tiles));
}


} // end experimental
} // end agency

