#include <agency/detail/config.hpp>
#include <agency/tuple/detail/tuple.hpp>
#include <utility>
#include <type_traits>


namespace agency
{


template<class... Types>
using tuple = agency::detail::tuple<Types...>;


template<class... Types>
__AGENCY_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<class... Types>
__AGENCY_ANNOTATION
tuple<typename std::decay<Types>::type...> make_tuple(Types&&... args)
{
  return tuple<typename std::decay<Types>::type...>(std::forward<Types>(args)...);
}


template<class... Types>
__AGENCY_ANNOTATION
tuple<Types&...> tie(Types&... args)
{
  return tuple<Types&...>(args...);
}


template<class... Args>
__AGENCY_ANNOTATION
tuple<Args&&...> forward_as_tuple(Args&&... args)
{
  return tuple<Args&&...>(std::forward<Args>(args)...);
}


} // end agency

