#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/tuple/detail/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if_not_scoped.hpp>

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
struct scoped_index
{
  using type = decltype(
    __tu::tuple_cat_apply(
      detail::index_tuple_maker{},
      detail::make_tuple_if_not_scoped<ExecutionCategory1>(std::declval<Index1>()),
      detail::make_tuple_if_not_scoped<ExecutionCategory2>(std::declval<Index2>())
    )
  );
};


template<class ExecutionCategory1,
         class ExecutionCategory2,
         class Index1,
         class Index2>
using scoped_index_t = typename scoped_index<
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
scoped_index_t<ExecutionCategory1,ExecutionCategory2,Index1,Index2> make_scoped_index(const Index1& outer_idx, const Index2& inner_idx)
{
  return __tu::tuple_cat_apply(
    detail::index_tuple_maker{},
    detail::make_tuple_if_not_scoped<ExecutionCategory1>(outer_idx),
    detail::make_tuple_if_not_scoped<ExecutionCategory2>(inner_idx)
  );
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
class tuple_size<agency::detail::index_tuple<Indices...>> : public std::tuple_size<agency::detail::tuple<Indices...>> {};

template<size_t i, class... Indices>
class tuple_element<i,agency::detail::index_tuple<Indices...>> : public std::tuple_element<i,agency::detail::tuple<Indices...>> {};


} // end namespace std

