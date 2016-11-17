#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/all.hpp>
#include <iterator>


namespace agency
{
namespace experimental
{


// XXX in c++17, the type of bound should be auto
template<class Range, std::size_t bound>
class statically_bounded_view
{
  private:
    using base_type = agency::experimental::all_t<Range>;

  public:
    using value_type = range_value_t<base_type>;
    using reference = range_reference_t<base_type>;
    using difference_type = range_difference_t<base_type>;

    // note the special size_type
    using size_type = bounded_size_t<bound>;

    __AGENCY_ANNOTATION
    statically_bounded_view() = default;

    __AGENCY_ANNOTATION
    statically_bounded_view(const statically_bounded_view&) = default;

    template<class OtherRange,
             __AGENCY_REQUIRES(
               std::is_constructible<
                 base_type,
                 all_t<OtherRange&&>
               >::value
             )>
    __AGENCY_ANNOTATION
    statically_bounded_view(OtherRange&& other)
      : base_(all(std::forward<OtherRange>(other)))
    {}

    __AGENCY_ANNOTATION
    static constexpr std::size_t static_bound = bound;

    __AGENCY_ANNOTATION
    static constexpr size_type max_size() const
    {
      return bounded_size_t<bound>(bound);
    }

    __AGENCY_ANNOTATION
    size_type size() const
    {
      return size_type(all_.size());
    }

    __AGENCY_ANNOTATION
    statically_bounded_view all() const
    {
      return *this;
    }

    __AGENCY_ANNOTATION
    reference operator()(size_type n) const
    {
      return base_[n.value()];
    }

  private:
    base_type base_;
};


template<class Range, std::size_t bound>
statically_bounded_view<Range,bound> all(const statically_bounded_view<Range,bound>& rng)
{
  return rng.all();
}


// XXX in c++17, the type of bound should be auto
template<std::size_t bound, class Range>
__AGENCY_ANNOTATION
statically_bounded_view<Range,bound> statically_bounded(Range&& rng)
{
  return statically_bounded_view<Range,bound>(std::forward<Range>(rng));
}


} // end experimental
} // end agency

