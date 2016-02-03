#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if_not_nested.hpp>

namespace agency
{
namespace detail
{


// index_tuple can't just be an alias for a particular kind of tuple
// because it also requires arithmetic operators
template<class... Indices>
class index_tuple :
  public agency::detail::tuple<Indices...>,
  public arithmetic_tuple_facade<index_tuple<Indices...>>
{
  public:
    using agency::detail::tuple<Indices...>::tuple;
};


template<class... Indices>
__AGENCY_ANNOTATION
index_tuple<Indices...> make_index_tuple(const std::tuple<Indices...>& indices)
{
  return index_tuple<Indices...>(indices);
}

template<class... Args>
__AGENCY_ANNOTATION
index_tuple<decay_t<Args>...> make_index_tuple(Args&&... args)
{
  return index_tuple<decay_t<Args>...>(std::forward<Args>(args)...);
}


struct index_tuple_maker
{
  template<class... Args>
  __AGENCY_ANNOTATION
  auto operator()(Args&&... args) const
    -> decltype(
         make_index_tuple(std::forward<Args>(args)...)
       )
  {
    return make_index_tuple(std::forward<Args>(args)...);
  }
};


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Index1,
         class Index2>
struct nested_index
{
  using type = decltype(
    __tu::tuple_cat_apply(
      detail::index_tuple_maker{},
      detail::make_tuple_if_not_nested<ExecutionCategory1>(std::declval<Index1>()),
      detail::make_tuple_if_not_nested<ExecutionCategory2>(std::declval<Index2>())
    )
  );
};


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Index1,
         class Index2>
using nested_index_t = typename nested_index<
  ExecutionCategory1,
  ExecutionCategory2,
  Index1,
  Index2
>::type;


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Index1,
         class Index2>
__AGENCY_ANNOTATION
nested_index_t<ExecutionCategory1,ExecutionCategory2,Index1,Index2> make_nested_index(const Index1& outer_idx, const Index2& inner_idx)
{
  return __tu::tuple_cat_apply(
    detail::index_tuple_maker{},
    detail::make_tuple_if_not_nested<ExecutionCategory1>(outer_idx),
    detail::make_tuple_if_not_nested<ExecutionCategory2>(inner_idx)
  );
}


// a point on the number line is bounded by another if it is strictly less than the other
template<class Integral1, class Integral2,
         class = typename std::enable_if<
           std::is_integral<Integral1>::value && std::is_integral<Integral2>::value
         >::type>
__AGENCY_ANNOTATION
bool is_bounded_by(const Integral1& x, const Integral2& bound)
{
  return x < bound;
}


// a multidimensional index is bounded by another if all of its elements are
// bounded by the corresponding element of the other
// essentially we test whether x is contained within (not lying on)
// the axis-aligned bounding box from the origin to the bound
template<class IndexTuple1, class IndexTuple2,
         class = typename std::enable_if<
           is_tuple<IndexTuple1>::value && is_tuple<IndexTuple2>::value
         >::type,
         class = typename std::enable_if<
           std::tuple_size<IndexTuple1>::value == std::tuple_size<IndexTuple2>::value
         >::type>
__AGENCY_ANNOTATION
bool is_bounded_by(const IndexTuple1& x, const IndexTuple2& bound);


// terminal case: x is bounded by bound
template<size_t i, class IndexTuple1, class IndexTuple2>
__AGENCY_ANNOTATION
typename std::enable_if<
  std::tuple_size<IndexTuple1>::value <= i,
  bool
>::type
  is_bounded_by_impl(const IndexTuple1& x, const IndexTuple2& bound)
{
  return true;
}


// recursive case: early out if x[i] is not bounded by bound[i]
template<size_t i, class IndexTuple1, class IndexTuple2>
__AGENCY_ANNOTATION
typename std::enable_if<
  i < std::tuple_size<IndexTuple1>::value,
  bool
>::type
  is_bounded_by_impl(const IndexTuple1& x, const IndexTuple2& bound)
{
  return detail::is_bounded_by(detail::get<i>(x), detail::get<i>(bound)) && detail::is_bounded_by_impl<i+1>(x,bound);
}


template<class IndexTuple1, class IndexTuple2,
         class EnableIf1, class EnableIf2>
__AGENCY_ANNOTATION
bool is_bounded_by(const IndexTuple1& x, const IndexTuple2& bound)
{
  return detail::is_bounded_by_impl<0>(x,bound);
}


} // end detail
} // end agency


namespace __tu
{

// tuple_traits specializations

template<class... Indices>
struct tuple_traits<agency::detail::index_tuple<Indices...>>
  : __tu::tuple_traits<agency::detail::tuple<Indices...>>
{
  using tuple_type = agency::detail::tuple<Indices...>;
}; // end tuple_traits


} // end __tu


namespace std
{


template<class... Indices>
struct tuple_size<agency::detail::index_tuple<Indices...>> : std::tuple_size<agency::detail::tuple<Indices...>> {};

template<size_t i, class... Indices>
struct tuple_element<i,agency::detail::index_tuple<Indices...>> : std::tuple_element<i,agency::detail::tuple<Indices...>> {};


} // end namespace std

