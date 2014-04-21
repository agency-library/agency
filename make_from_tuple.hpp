#include <tuple>
#include <integer_sequence>
#include <type_traits>


template<typename T, typename Tuple, size_t... I>
T __make_from_tuple(Tuple&& t, std::index_sequence<I...>)
{
  return T(std::get<I>(std::forward<Tuple>(t))...);
}


template<typename T, typename Tuple>
T make_from_tuple(Tuple&& t)
{
  using indices = std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>;
  return __make_from_tuple<T>(std::forward<Tuple>(t), indices{});
}

