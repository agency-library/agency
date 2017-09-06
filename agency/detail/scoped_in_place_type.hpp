#pragma once

#include <agency/detail/config.hpp>
#include <agency/experimental/variant.hpp>


namespace agency
{
namespace detail
{


template<class OuterType, class... InnerTypes>
struct scoped_in_place_type_t
{
  using outer_type = agency::experimental::in_place_type_t<OuterType>;

  using inner_type = scoped_in_place_type_t<InnerTypes...>;

  __AGENCY_ANNOTATION
  outer_type outer() const
  {
    return outer_type();
  }

  __AGENCY_ANNOTATION
  inner_type inner() const
  {
    return inner_type();
  }
};

template<class OuterType>
struct scoped_in_place_type_t<OuterType>
{
  using outer_type = agency::experimental::in_place_type_t<OuterType>;

  __AGENCY_ANNOTATION
  outer_type outer() const
  {
    return outer_type();
  }
};


template<class OuterType, class... InnerTypes>
struct make_scoped_in_place_type_t_impl
{
  using type = scoped_in_place_type_t<OuterType, InnerTypes...>;
};

// when a single type is given, and that type is already a scoped_in_place_type_t, don't nest it
// instead, just act like the identity function
template<class OuterType, class... InnerTypes>
struct make_scoped_in_place_type_t_impl<scoped_in_place_type_t<OuterType, InnerTypes...>>
{
  using type = scoped_in_place_type_t<OuterType, InnerTypes...>;
};

template<class OuterType, class... InnerTypes>
using make_scoped_in_place_type_t = typename make_scoped_in_place_type_t_impl<OuterType,InnerTypes...>::type;


// scoped_in_place_type_t_cat is like tuple_cat for scoped_in_place_type_t
template<class ScopedInPlaceType, class... ScopedInPlaceTypes>
struct scoped_in_place_type_t_cat;

// for a single scoped_in_place_type_t, scoped_in_place_type_t_cat is the identity
template<class... Types>
struct scoped_in_place_type_t_cat<scoped_in_place_type_t<Types...>>
{
  using type = scoped_in_place_type_t<Types...>;
};

// for two scoped_in_place_type_t, combine the two lists of types
template<class... Types1, class... Types2>
struct scoped_in_place_type_t_cat<scoped_in_place_type_t<Types1...>, scoped_in_place_type_t<Types2...>>
{
  using type = scoped_in_place_type_t<Types1..., Types2...>;
};

// for multiple scoped_in_place_type_t, recurse twice
template<class... Types1, class... Types2, class... ScopedInPlaceTypes>
struct scoped_in_place_type_t_cat<scoped_in_place_type_t<Types1...>, scoped_in_place_type_t<Types2...>, ScopedInPlaceTypes...>
{
  using type = typename scoped_in_place_type_t_cat<
    scoped_in_place_type_t<Types1...>,
    typename scoped_in_place_type_t_cat<
      scoped_in_place_type_t<Types2...>,
      ScopedInPlaceTypes...
    >::type
  >::type;
};


template<class ScopedInPlaceType, class... ScopedInPlaceTypes>
using scoped_in_place_type_t_cat_t = typename scoped_in_place_type_t_cat<ScopedInPlaceType, ScopedInPlaceTypes...>::type;


} // end detail
} // end agency


