#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple/detail/tuple_utility.hpp>
#include <tuple>
#include <utility>
#include <type_traits>


// XXX move this content up into agency/tuple.hpp
//     there's not really a need to reserve a separate directory for tuple


namespace agency
{


template<class... Types>
using tuple = __tu::tuple<Types...>;


namespace detail
{


struct ignore_t
{
  template<class T>
  __AGENCY_ANNOTATION
  const ignore_t operator=(T&&) const
  {
    return *this;
  }
};


} // end detail


constexpr detail::ignore_t ignore{};


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


namespace detail
{


template<class IndexSequence, class... Tuples>
struct tuple_cat_result_impl_impl;


template<size_t... I, class... Tuples>
struct tuple_cat_result_impl_impl<index_sequence<I...>, Tuples...>
{
  using type = tuple<typename __tu::__tuple_cat_element<I, Tuples...>::type...>;
};


template<class... Tuples>
struct tuple_cat_result_impl
{
  static const size_t result_size = __tu::__sum<0u, std::tuple_size<Tuples>::value...>::value;

  using type = typename tuple_cat_result_impl_impl<
    make_index_sequence<result_size>,
    Tuples...
  >::type;
};


template<class... Tuples>
using tuple_cat_result = typename tuple_cat_result_impl<typename std::decay<Tuples>::type...>::type;


template<class T>
struct maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  T operator()(Args&&... args)
  {
    return T{std::forward<Args>(args)...};
  }
};


// XXX this should be moved into some file underneath agency/detail/tuple
template<class T, class Tuple>
__AGENCY_ANNOTATION
T make_from_tail(Tuple&& t)
{
  return __tu::tuple_tail_invoke(std::forward<Tuple>(t), detail::maker<T>());
}


} // end detail


// XXX this doesn't forward tuple elements which are reference types correctly
//     because make_tuple() doesn't do that
template<class... Tuples>
__AGENCY_ANNOTATION
detail::tuple_cat_result<Tuples...> tuple_cat(Tuples&&... tuples)
{
  return __tu::tuple_cat_apply(detail::maker<detail::tuple_cat_result<Tuples...>>{}, std::forward<Tuples>(tuples)...);
}


// an output operator for tuple
template<class... Types>
std::ostream& operator<<(std::ostream& os, const tuple<Types...>& t)
{
  os << "{";
  __tu::tuple_print(t, os);
  os << "}";
  return os;
}


} // end agency

