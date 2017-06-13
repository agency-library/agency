#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/untile.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <agency/container/array.hpp>
#include <agency/container/vector.hpp>
#include <memory>

namespace agency
{
namespace experimental
{


// XXX until this thing can resize, call it an array
// XXX probably want to take a single allocator as a parameter instead of two separate allocators
template<class T, template<class> class InnerAlloc = allocator, template<class> class OuterAlloc = allocator>
class tiled_array
{
  public:
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;

  private:
    size_type tile_size_;

    using inner_allocator_type = InnerAlloc<T>;
    using inner_container = vector<value_type, inner_allocator_type>;

    using outer_allocator_type = OuterAlloc<inner_container>;
    using outer_container = vector<inner_container, outer_allocator_type>;

    outer_container tiles_;

  public:
    tiled_array() : tile_size_(0), tiles_() {}

    tiled_array(const tiled_array&) = default;

    // constructs a tiled_array with a tile for each allocator
    template<class Range,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 inner_allocator_type,
                 range_value_t<Range>
               >::value
             )>
    tiled_array(size_type n,
                const value_type& val,
                const Range& allocators)
    {
      size_type num_tiles = allocators.size();

      assert(num_tiles > 0 && num_tiles <= all_t::max_tile_count);

      if(n >= allocators.size())
      {
        // equally distribute allocations
        tile_size_ = (n + num_tiles - 1) / num_tiles;
        size_type last_tile_size = n - (tile_size_ * (allocators.size() - 1));

        // construct the full tiles
        auto last_alloc = --allocators.end();
        for(auto alloc = allocators.begin(); alloc != last_alloc; ++alloc)
        {
          tiles_.emplace_back(tile_size_, val, *alloc);
        }

        // construct the last tile
        tiles_.emplace_back(last_tile_size, val, *last_alloc);
      }
      else
      {
        // when there are more allocators than there are elements,
        // just create a tile for each element
        // an alternative would be to create a single tile using the first allocator
        // or choose some minimum tile size
        // we might want a constructor which accepts a tile_size
        tile_size_ = 1;
        auto alloc = allocators.begin();
        for(size_type i = 0; i < n; ++i, ++alloc)
        {
          tiles_.emplace_back(tile_size_, val, *alloc);
        }
      }
    }

    // constructs a tiled_array with a single tile
    tiled_array(size_type n, const value_type& val = value_type())
      : tiled_array(n, val, array<inner_allocator_type,1>{inner_allocator_type()})
    {}

    size_type tile_size() const
    {
      return tile_size_;
    }

  public:
    using all_t = small_untiled_view<outer_container>;

    all_t all()
    {
      return untile(tile_size(), tiles_);
    }

    using const_all_t = small_untiled_view<const outer_container>;

    const_all_t all() const
    {
      return untile(tile_size(), tiles_);
    }

    size_type size() const
    {
      return all().size();
    }

    using iterator = range_iterator_t<all_t>;

    iterator begin()
    {
      return all().begin();
    }

    iterator end()
    {
      return all().end();
    }

    using const_iterator = range_iterator_t<const_all_t>;

    const_iterator begin() const
    {
      return all().begin();
    }

    const_iterator end() const
    {
      return all().end();
    }

    using tiles_view = decltype(agency::experimental::all(std::declval<const outer_container&>()));

    tiles_view tiles() const
    {
      return agency::experimental::all(tiles_);
    }

    const inner_container& tile(size_type i) const
    {
      return tiles()[i];
    }

    reference operator[](size_type i)
    {
      return all()[i];
    }

    const_reference operator[](size_type i) const
    {
      return all()[i];
    }

    void clear()
    {
      tiles_.clear();
    }

    bool operator==(const tiled_array& rhs) const
    {
      return tiles_ == rhs.tiles_;
    }
};


} // end experimental
} // end agency

