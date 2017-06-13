#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/flatten.hpp>
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
class segmented_array
{
  private:
    using inner_allocator_type = InnerAlloc<T>;

  public:
    using value_type = T;
    using reference = value_type&;
    using const_reference = const value_type&;
    using size_type = std::size_t;

    segmented_array() = default;

    segmented_array(const segmented_array&) = default;

    // constructs a segmented_array with a segment for each allocator
    template<class Range,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 inner_allocator_type,
                 range_value_t<Range>
               >::value
             )>
    segmented_array(size_type n,
                    const value_type& val,
                    const Range& allocators)
    {
      size_type num_segments = allocators.size();
      assert(num_segments > 0);

      if(n >= allocators.size())
      {
        // equally distribute allocations
        size_type segment_size = (n + num_segments - 1) / num_segments;
        size_type last_segment_size = n - (segment_size * (allocators.size() - 1));

        // construct the full segments
        auto last_alloc = --allocators.end();
        for(auto alloc = allocators.begin(); alloc != last_alloc; ++alloc)
        {
          segments_.emplace_back(segment_size, val, *alloc);
        }

        // construct the last segment
        segments_.emplace_back(last_segment_size, val, *last_alloc);
      }
      else
      {
        // when there are more allocators than there are elements,
        // just create a segment for each element
        // an alternative would be to create a single segment using the first allocator
        // or choose some minimum segment size
        // we might want a constructor which accepts a segment_size
        auto alloc = allocators.begin();
        for(size_type i = 0; i < n; ++i, ++alloc)
        {
          segments_.emplace_back(1, val, *alloc);
        }
      }
    }

    // constructs a segmented_array with a single segment
    segmented_array(size_type n, const value_type& val = value_type())
      : segmented_array(n, val, array<inner_allocator_type,1>{inner_allocator_type()})
    {}

  private:
    using inner_container = vector<value_type, inner_allocator_type>;

    // XXX will want to parameterize the outer allocator used to store the segments
    using outer_allocator_type = OuterAlloc<inner_container>;
    using outer_container = vector<inner_container, outer_allocator_type>;
    outer_container segments_;

  public:
    using all_t = flatten_view<outer_container>;

    all_t all()
    {
      return flatten(segments_);
    }

    using const_all_t = flatten_view<const outer_container>;

    const_all_t all() const
    {
      return flatten(segments_);
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

    using segments_view = decltype(agency::experimental::all(std::declval<const outer_container&>()));

    segments_view segments() const
    {
      return agency::experimental::all(segments_);
    }

    const inner_container& segment(size_type i) const
    {
      return segments()[i];
    }

    using segment_iterator = range_iterator_t<segments_view>;

    segment_iterator segments_begin() const
    {
      return segments().begin();
    }

    segment_iterator segments_end() const
    {
      return segments().end();
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
      segments_.clear();
    }

    bool operator==(const segmented_array& rhs) const
    {
      return segments_ == rhs.segments_;
    }
};


} // end experimental
} // end agency

