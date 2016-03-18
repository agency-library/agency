#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/span.hpp>

namespace agency
{
namespace experimental
{


template<class Container>
__AGENCY_ANNOTATION
span<typename Container::value_type> view(Container& c)
{
  return span<typename Container::value_type>(c);
}

template<class Container>
__AGENCY_ANNOTATION
span<const typename Container::value_type> view(const Container& c)
{
  return span<const typename Container::value_type>(c);
}


} // end experimental
} // end agency

