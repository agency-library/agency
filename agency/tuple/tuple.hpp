#include <agency/detail/config.hpp>
#include <agency/tuple/detail/tuple.hpp>

namespace agency
{

template<class... Types>
using tuple = agency::detail::tuple<Types...>;

using __tu::swap;
using __tu::make_tuple;
using __tu::tie;
using __tu::forward_as_tuple;

} // end agency

