#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/ranges/zip_with.hpp>

namespace agency
{
namespace experimental
{



template<class Function, class... Ranges>
using transformed_view = zip_with_view<Function, Ranges...>;


// transformed() is a synonym for zip_with()
// it's named transformed rather than transform to avoid collision with
// a potential transform algorithm
template<class Function, class... Ranges>
__AGENCY_ANNOTATION
transformed_view<Function,Ranges...> transformed(Function f, Ranges&&... ranges)
{
  return transformed_view<Function,Ranges...>(f, std::forward<Ranges>(ranges)...);
}


} // end experimental
} // end agency

