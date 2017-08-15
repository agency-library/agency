#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/range_traits.hpp>
#include <agency/experimental/ranges/zip_with.hpp>
#include <type_traits>
#include <iterator>

namespace agency
{
namespace experimental
{
namespace detail
{


struct forward_as_tuple_functor
{
  template<class... Args>
  __AGENCY_ANNOTATION
  agency::tuple<Args&&...> operator()(Args&&... args) const
  {
    return agency::forward_as_tuple(std::forward<Args>(args)...);
  }
};

} // end detail


template<class Range, class... Ranges>
class zip_view : zip_with_view<detail::forward_as_tuple_functor, Range, Ranges...>
{
  private:
    using super_t = zip_with_view<detail::forward_as_tuple_functor, Range, Ranges...>;

  public:
    using iterator = typename super_t::iterator;
    using sentinel = typename super_t::sentinel;

    template<class OtherRange, class... OtherRanges>
    __AGENCY_ANNOTATION
    zip_view(OtherRange&& rng, OtherRanges&&... rngs)
      : super_t(detail::forward_as_tuple_functor{}, std::forward<OtherRange>(rng), std::forward<OtherRanges>(rngs)...)
    {}

    using super_t::super_t;
    using super_t::begin;
    using super_t::end;
    using super_t::operator[];
    using super_t::size;
};


template<class Range, class... Ranges>
__AGENCY_ANNOTATION
zip_view<Range,Ranges...> zip(Range&& rng, Ranges&&... ranges)
{
  return zip_view<Range,Ranges...>(std::forward<Range>(rng), std::forward<Ranges>(ranges)...);
}


} // end experimental
} // end agency

