#include <agency/detail/config.hpp>
#include <agency/tuple/detail/tuple.hpp>

namespace agency
{

template<class... Types>
using tuple = agency::detail::tuple<Types...>;

} // end agency

