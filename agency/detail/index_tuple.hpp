#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/tuple/arithmetic_tuple_facade.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/make_tuple_if.hpp>


namespace agency
{
namespace detail
{


// index_tuple can't just be an alias for a particular kind of tuple
// because it also requires arithmetic operators
template<class... Indices>
class index_tuple :
  public agency::tuple<Indices...>,
  public arithmetic_tuple_facade<index_tuple<Indices...>>
{
  using super_t = agency::tuple<Indices...>;

  public:
    // XXX workaround nvbug 2316472
    //using super_t::super_t;
    template<class... Args,
             __AGENCY_REQUIRES(
               std::is_constructible<super_t, Args&&...>::value
             )>
    __AGENCY_ANNOTATION
    index_tuple(Args&&... args)
      : super_t(std::forward<Args>(args)...)
    {}
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


template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterIndex,
         class InnerIndex>
struct scoped_index
{
  using type = decltype(
    __tu::tuple_cat_apply(
      detail::index_tuple_maker{},
      detail::make_tuple_if<OuterDepth == 1>(std::declval<OuterIndex>()),
      detail::make_tuple_if<InnerDepth == 1>(std::declval<InnerIndex>())
    )
  );
};


template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterIndex,
         class InnerIndex>
using scoped_index_t = typename scoped_index<
  OuterDepth,
  InnerDepth,
  OuterIndex,
  InnerIndex
>::type;


template<size_t OuterDepth,
         size_t InnerDepth,
         class OuterIndex,
         class InnerIndex>
__AGENCY_ANNOTATION
scoped_index_t<OuterDepth,InnerDepth,OuterIndex,InnerIndex> make_scoped_index(const OuterIndex& outer_idx, const InnerIndex& inner_idx)
{
  return __tu::tuple_cat_apply(
    detail::index_tuple_maker{},
    detail::make_tuple_if<OuterDepth == 1>(outer_idx),
    detail::make_tuple_if<InnerDepth == 1>(inner_idx)
  );
}


} // end detail
} // end agency


namespace std
{


template<class... Indices>
class tuple_size<agency::detail::index_tuple<Indices...>> : public std::tuple_size<agency::tuple<Indices...>> {};

template<size_t i, class... Indices>
class tuple_element<i,agency::detail::index_tuple<Indices...>> : public std::tuple_element<i,agency::tuple<Indices...>> {};


} // end namespace std

