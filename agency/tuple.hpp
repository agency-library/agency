#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/tuple/tuple_base.hpp>
#include <agency/detail/tuple/get.hpp>
#include <tuple>
#include <utility>
#include <type_traits>
#include <iostream>


namespace agency
{


template<class... Types>
class tuple
{
  private:
    using base_type = detail::tuple_base<detail::make_index_sequence<sizeof...(Types)>, Types...>;
    base_type base_;

    __AGENCY_ANNOTATION
    base_type& base()
    {
      return base_;
    }

    __AGENCY_ANNOTATION
    const base_type& base() const
    {
      return base_;
    }

  public:
    __AGENCY_ANNOTATION
    tuple() : base_{} {};

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               detail::conjunction<
                 std::is_constructible<Types,UTypes&&>...
               >::value &&
               detail::conjunction<
                 std::is_convertible<UTypes&&,Types>...
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple(UTypes&&... args)
      : base_{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               detail::conjunction<
                 std::is_constructible<Types,UTypes&&>...
               >::value &&
               not
               detail::disjunction<
                 std::is_convertible<UTypes&&,Types>...
               >::value
             )>
    __AGENCY_ANNOTATION
    explicit tuple(UTypes&&... args)
      : base_{std::forward<UTypes>(args)...}
    {}

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 detail::conjunction<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             )>
    __AGENCY_ANNOTATION
    tuple(const tuple<UTypes...>& other)
      : base_{other.base()}
    {}

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 detail::conjunction<
                   std::is_constructible<Types,UTypes&&>...
                 >::value
             )>
    __AGENCY_ANNOTATION
    tuple(tuple<UTypes...>&& other)
      : base_{std::move(other.base())}
    {}

    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               detail::conjunction<
                 std::is_constructible<typename std::tuple_element<                            0,tuple>::type,const UType1&>,
                 std::is_constructible<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple>::type,const UType2&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple(const std::pair<UType1,UType2>& p)
      : base_{p.first, p.second}
    {}

    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               detail::conjunction<
                 std::is_constructible<typename std::tuple_element<                            0,tuple>::type,UType1&&>,
                 std::is_constructible<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple>::type,UType2&&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple(std::pair<UType1,UType2>&& p)
      : base_{std::move(p.first), std::move(p.second)}
    {}

    __AGENCY_ANNOTATION
    tuple(const tuple& other)
      : base_{other.base()}
    {}

    __AGENCY_ANNOTATION
    tuple(tuple&& other)
      : base_{std::move(other.base())}
    {}

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
                 detail::conjunction<
                   std::is_constructible<Types,const UTypes&>...
                 >::value
             )>
    __AGENCY_ANNOTATION
    tuple(const std::tuple<UTypes...>& other)
      : base_{other}
    {}

    __AGENCY_ANNOTATION
    tuple& operator=(const tuple& other)
    {
      base().operator=(other.base());
      return *this;
    }

    __AGENCY_ANNOTATION
    tuple& operator=(tuple&& other)
    {
      base().operator=(std::move(other.base()));
      return *this;
    }

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) and
               detail::conjunction<
                 std::is_assignable<Types, const UTypes&>...
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple& operator=(const tuple<UTypes...>& other)
    {
      base().operator=(other.base());
      return *this;
    }

    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) and
               detail::conjunction<
                 std::is_assignable<Types, UTypes&&>...
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple& operator=(tuple<UTypes...>&& other)
    {
      base().operator=(std::move(other.base()));
      return *this;
    }

    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               detail::conjunction<
                 std::is_assignable<typename std::tuple_element<                            0,tuple>::type,const UType1&>,
                 std::is_assignable<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple>::type,const UType2&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple& operator=(const std::pair<UType1,UType2>& p)
    {
      base().operator=(p);
      return *this;
    }

    template<class UType1, class UType2,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == 2) &&
               detail::conjunction<
                 std::is_assignable<typename std::tuple_element<                            0,tuple>::type,UType1&&>,
                 std::is_assignable<typename std::tuple_element<sizeof...(Types) == 2 ? 1 : 0,tuple>::type,UType2&&>
               >::value
             )>
    __AGENCY_ANNOTATION
    tuple& operator=(std::pair<UType1,UType2>&& p)
    {
      base().operator=(std::move(p));
      return *this;
    }

    __AGENCY_ANNOTATION
    void swap(tuple& other)
    {
      base().swap(other.base());
    }

    // enable conversion to Tuple-like things
    // note that this is non-standard
    // XXX enable conversion to Tuple when Tuple is_tuple
    template<class... UTypes,
             __AGENCY_REQUIRES(
               (sizeof...(Types) == sizeof...(UTypes)) &&
               detail::conjunction<
                 std::is_constructible<Types,const UTypes&>...
                >::value
             )>
    __AGENCY_ANNOTATION
    operator std::tuple<UTypes...> () const
    {
      return static_cast<std::tuple<UTypes...>>(base());
    }

    // mutable member get
    // note that this is non-standard
    template<size_t i, __AGENCY_REQUIRES(i < std::tuple_size<tuple>::value)>
    __AGENCY_ANNOTATION
    typename std::tuple_element<i,tuple>::type& get() &
    {
      return base().template mutable_get<i>();
    }

    // const member get
    // note that this is non-standard
    template<size_t i, __AGENCY_REQUIRES(i < std::tuple_size<tuple>::value)>
    __AGENCY_ANNOTATION
    const typename std::tuple_element<i,tuple>::type& get() const &
    {
      return base().template const_get<i>();
    }

    // moving member get
    // note that this is non-standard
    template<size_t i, __AGENCY_REQUIRES(i < std::tuple_size<tuple>::value)>
    __AGENCY_ANNOTATION
    typename std::tuple_element<i,tuple>::type&& get() &&
    {
      using type = typename std::tuple_element<i, tuple>::type;
  
      auto&& leaf = static_cast<agency::detail::tuple_leaf<i,type>&&>(base());
  
      return static_cast<type&&>(leaf.mutable_get());
    }

  private:
    template<class... UTypes>
    friend class tuple;
};


template<>
class tuple<>
{
  public:
    __AGENCY_ANNOTATION
    void swap(tuple&){}
};


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


template<class... Types>
__AGENCY_ANNOTATION
void swap(tuple<Types...>& a, tuple<Types...>& b)
{
  a.swap(b);
}


template<size_t i, class... UTypes>
__AGENCY_ANNOTATION
typename std::tuple_element<i, tuple<UTypes...>>::type &
  get(tuple<UTypes...>& t)
{
  return t.template get<i>();
}


template<size_t i, class... UTypes>
__AGENCY_ANNOTATION
const typename std::tuple_element<i, tuple<UTypes...>>::type &
  get(const tuple<UTypes...>& t)
{
  return t.template get<i>();
}


template<size_t i, class... UTypes>
__AGENCY_ANNOTATION
typename std::tuple_element<i, tuple<UTypes...>>::type &&
  get(tuple<UTypes...>&& t)
{
  return std::move(t).template get<i>();
}


namespace detail
{
namespace tuple_detail
{


template<size_t I, class T, class... Types>
struct find_exactly_one_impl;


template<size_t I, class T, class U, class... Types>
struct find_exactly_one_impl<I,T,U,Types...> : find_exactly_one_impl<I+1, T, Types...> {};


template<size_t I, class T, class... Types>
struct find_exactly_one_impl<I,T,T,Types...> : std::integral_constant<size_t, I>
{
  static_assert(find_exactly_one_impl<I,T,Types...>::value == -1, "type can only occur once in type list");
};


template<size_t I, class T>
struct find_exactly_one_impl<I,T> : std::integral_constant<int, -1> {};


template<class T, class... Types>
struct find_exactly_one : find_exactly_one_impl<0,T,Types...>
{
  static_assert(int(find_exactly_one::value) != -1, "type not found in type list");
};


} // end tuple_detail
} // end detail


template<class T, class... Types>
__AGENCY_ANNOTATION
T& get(tuple<Types...>& t)
{
  return agency::get<agency::detail::tuple_detail::find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__AGENCY_ANNOTATION
const T& get(const tuple<Types...>& t)
{
  return agency::get<agency::detail::tuple_detail::find_exactly_one<T,Types...>::value>(t);
}


template<class T, class... Types>
__AGENCY_ANNOTATION
T&& get(tuple<Types...>&& t)
{
  return agency::get<agency::detail::tuple_detail::find_exactly_one<T,Types...>::value>(std::move(t));
}


namespace detail
{
namespace tuple_detail
{


// declare these helper functions for operator== and operator<


template<class... TTypes, class... UTypes>
__AGENCY_ANNOTATION
bool equal(const tuple<TTypes...>&, const tuple<UTypes...>&, index_sequence<>);

template<class... TTypes, class... UTypes, size_t I, size_t... Is>
__AGENCY_ANNOTATION
bool equal(const tuple<TTypes...>&, const tuple<UTypes...>&, index_sequence<I, Is...>);

__agency_exec_check_disable__
template<class... TTypes, class... UTypes>
__AGENCY_ANNOTATION
bool less_than(const tuple<TTypes...>& t, const tuple<UTypes...>& u, index_sequence<>);

__agency_exec_check_disable__
template<class... TTypes, class... UTypes, size_t I, size_t... Is>
__AGENCY_ANNOTATION
bool less_than(const tuple<TTypes...>& t, const tuple<UTypes...>& u, index_sequence<I, Is...>);


} // end tuple_detail
} // end detail


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator==(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return detail::tuple_detail::equal(t, u, detail::make_index_sequence<sizeof...(TTypes)>{});
}


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator<(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return detail::tuple_detail::less_than(t, u, detail::make_index_sequence<sizeof...(TTypes)>{});
}


namespace detail
{
namespace tuple_detail 
{


// these definitions must come after operator== and operator< because they use those operators when their parameters are nested tuples


template<class... TTypes, class... UTypes>
__AGENCY_ANNOTATION
bool equal(const tuple<TTypes...>&, const tuple<UTypes...>&, index_sequence<>)
{
  return true;
}


__agency_exec_check_disable__
template<class... TTypes, class... UTypes, size_t I, size_t... Is>
__AGENCY_ANNOTATION
bool equal(const tuple<TTypes...>& t, const tuple<UTypes...>& u, index_sequence<I, Is...>)
{
  return (agency::get<I>(t) == agency::get<I>(u)) && tuple_detail::equal(t, u, index_sequence<Is...>());
}


template<class... TTypes, class... UTypes>
__AGENCY_ANNOTATION
bool less_than(const tuple<TTypes...>&, const tuple<UTypes...>&, index_sequence<>)
{
  return false;
}


__agency_exec_check_disable__
template<class... TTypes, class... UTypes, size_t I, size_t... Is>
__AGENCY_ANNOTATION
bool less_than(const tuple<TTypes...>& t, const tuple<UTypes...>& u, index_sequence<I, Is...>)
{
  return (bool)(agency::get<0>(t) < agency::get<0>(u)) || (!(bool)(agency::get<0>(t) < agency::get<0>(u)) && tuple_detail::less_than(t, u, index_sequence<Is...>()));
}


} // end tuple_detail
} // end detail


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator!=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t == u);
}


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator>(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return u < t;
}


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator<=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(u < t);
}


template<class... TTypes, class... UTypes, __AGENCY_REQUIRES(sizeof...(TTypes) == sizeof...(UTypes))>
__AGENCY_ANNOTATION
bool operator>=(const tuple<TTypes...>& t, const tuple<UTypes...>& u)
{
  return !(t < u);
}


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


// declare tuple's operator<< before tuple_print, which may use it below
template<class... Types>
std::ostream& operator<<(std::ostream& os, const tuple<Types...>& t);


namespace detail
{
namespace tuple_detail
{


template<size_t i,
         class Tuple, class T,
         __AGENCY_REQUIRES(
           std::tuple_size<Tuple>::value > 0 &&
           i == (std::tuple_size<Tuple>::value - 1)
         )>
void output(std::ostream& os, const Tuple& t, const T&)
{
  // omit the delimiter for the final element
  os << agency::get<i>(t);
}


template<size_t i,
         class Tuple, class T,
         __AGENCY_REQUIRES(
           std::tuple_size<Tuple>::value > 0 &&
           i != (std::tuple_size<Tuple>::value - 1)
         )>
void output(std::ostream& os, const Tuple& t, const T& delimiter)
{
  os << agency::get<i>(t) << delimiter;

  tuple_detail::output<i+1>(os, t, delimiter);
}


template<size_t i,
         class Tuple, class T,
         __AGENCY_REQUIRES(
           std::tuple_size<Tuple>::value == 0
         )>
void output(std::ostream&, const Tuple&, const T&)
{
  // output nothing for zero-sized tuples
}


} // end tuple_detail
} // end detail


// note that this operator<< for tuple is non-standard
template<class... Types>
std::ostream& operator<<(std::ostream& os, const tuple<Types...>& t)
{
  os << "{";
  detail::tuple_detail::output<0>(os, t, ", ");
  os << "}";
  return os;
}


// a fancy version of get<i>() which works for Tuple-like types which
// either have
//
//   1. A .get<i>() member function or
//   2. An overloaded std::get<i>() function or
//   3. operator[]
template<std::size_t i, class Tuple,
         __AGENCY_REQUIRES(
           detail::tuple_detail::has_get_member_function<Tuple&&,i>::value or
           detail::tuple_detail::has_std_get_free_function<Tuple&&,i>::value or
           detail::tuple_detail::has_operator_bracket<Tuple&&>::value
        )>
__AGENCY_ANNOTATION
auto get(Tuple&& t) ->
  decltype(detail::tuple_detail::get<i>(std::forward<Tuple>(t)))
{
  return detail::tuple_detail::get<i>(std::forward<Tuple>(t));
}


namespace detail
{


// tuple_get_result computes the return type of std::get<i>(tuple)
template<size_t i, class TupleReference>
struct tuple_get_result
{
  using type = typename propagate_reference<
    TupleReference,
    typename std::tuple_element<
      i,
      typename std::decay<TupleReference>::type
    >::type
  >::type;
};

template<size_t i, class TupleReference>
using tuple_get_result_t = typename tuple_get_result<i,TupleReference>::type;



// tuple_cat_element computes the type of the ith element of the tuple returned by tuple_cat()
template<std::size_t i, class... Tuples>
struct tuple_cat_element;

template<std::size_t i, class Tuple1, class... Tuples>
struct tuple_cat_element<i, Tuple1, Tuples...>
{
  static const size_t size1 = std::tuple_size<Tuple1>::value;

  using type = typename lazy_conditional<
    (i < size1),
    std::tuple_element<i,Tuple1>,
    tuple_cat_element<i - size1, Tuples...>
  >::type;
};

template<std::size_t i, class Tuple>
struct tuple_cat_element<i,Tuple> : std::tuple_element<i,Tuple> {};

template<std::size_t i, class... Tuples>
using tuple_cat_element_t = typename tuple_cat_element<i,Tuples...>::type;



// tuple_cat_size computes the size of the tuple returned by tuple_cat()
template<class... TupleReferences>
struct tuple_cat_size;

template<>
struct tuple_cat_size<> : std::integral_constant<std::size_t, 0> {};

template<class TupleReference1, class... TupleReferences>
struct tuple_cat_size<TupleReference1, TupleReferences...>
  : std::integral_constant<
      std::size_t,
      std::tuple_size<typename std::decay<TupleReference1>::type>::value + tuple_cat_size<TupleReferences...>::value
    >
{};



// tuple_cat_get_result computes the type of tuple_cat_get()'s result
template<size_t i, class... Tuples>
struct tuple_cat_get_result;

template<std::size_t i, class TupleReference1, class... TupleReferences>
struct tuple_cat_get_result<i, TupleReference1, TupleReferences...>
{
  static const size_t size1 = std::tuple_size<typename std::decay<TupleReference1>::type>::value;

  using type = typename lazy_conditional<
    (i < size1),
    tuple_get_result<i,TupleReference1>,
    tuple_cat_get_result<i - size1, TupleReferences...>
  >::type;
};

template<std::size_t i, class TupleReference>
struct tuple_cat_get_result<i,TupleReference> : tuple_get_result<i,TupleReference> {};

template<std::size_t i, class... TupleReferences>
using tuple_cat_get_result_t = typename tuple_cat_get_result<i,TupleReferences...>::type;



// tuple_cat_get() returns the ith element of the tuple which would be returned by tuple_cat() as applied to these tuple parameters
template<std::size_t i, class Tuple1, class... Tuples>
__AGENCY_ANNOTATION
tuple_cat_get_result_t<i,Tuple1&&,Tuples&&...> tuple_cat_get(Tuple1&& tuple1, Tuples&&... tuples);

// when the element is in the first tuple, just use std::get<i>()
__agency_exec_check_disable__
template<std::size_t i, class Tuple1, class... Tuples,
         __AGENCY_REQUIRES(
           i < std::tuple_size<typename std::decay<Tuple1>::type>::value
        )>
__AGENCY_ANNOTATION
tuple_cat_get_result_t<i,Tuple1&&,Tuples&&...> tuple_cat_get_impl(Tuple1&& tuple1, Tuples&&...)
{
  return agency::get<i>(std::forward<Tuple1>(tuple1));
}


// when the element is not in the first tuple, recurse with tuple_cat_get()
template<std::size_t i, class Tuple1, class... Tuples,
         __AGENCY_REQUIRES(
           i >= std::tuple_size<typename std::decay<Tuple1>::type>::value
        )>
__AGENCY_ANNOTATION
tuple_cat_get_result_t<i,Tuple1&&,Tuples&&...> tuple_cat_get_impl(Tuple1&&, Tuples&&... tuples)
{
  const size_t j = i - std::tuple_size<typename std::decay<Tuple1>::type>::value;
  return detail::tuple_cat_get<j>(std::forward<Tuples>(tuples)...);
}

template<std::size_t i, class Tuple1, class... Tuples>
__AGENCY_ANNOTATION
tuple_cat_get_result_t<i,Tuple1&&,Tuples&&...> tuple_cat_get(Tuple1&& tuple1, Tuples&&... tuples)
{
  return detail::tuple_cat_get_impl<i>(std::forward<Tuple1>(tuple1), std::forward<Tuples>(tuples)...);
}



// tuple_cat_result computes the type of tuple returned by tuple_cat()
template<class IndexSequence, class... TupleReferences>
struct tuple_cat_result_impl;

template<std::size_t... indices, class... TupleReferences>
struct tuple_cat_result_impl<index_sequence<indices...>, TupleReferences...>
{
  using type = tuple<tuple_cat_element_t<indices, typename std::decay<TupleReferences>::type...>...>;
};

// XXX consider making the first parameter a template template parameter defining the kind of Tuple to return
//     i.e. template<class...> class TupleLike
template<class... TupleReferences>
struct tuple_cat_result
{
  // compute the size of the tuple we will return
  static const std::size_t size = tuple_cat_size<typename std::decay<TupleReferences>::type...>::value;

  // make an index sequence of that size and call the implementation
  using type = typename tuple_cat_result_impl<make_index_sequence<size>, TupleReferences...>::type;
};

template<class... TupleReferences>
using tuple_cat_result_t = typename tuple_cat_result<TupleReferences...>::type;



__agency_exec_check_disable__
template<size_t... indices, class... Tuples>
__AGENCY_ANNOTATION
tuple_cat_result_t<Tuples&&...> tuple_cat_impl(index_sequence<indices...>, Tuples&&... tuples)
{
  return detail::tuple_cat_result_t<Tuples&&...>(tuple_cat_get<indices>(std::forward<Tuples>(tuples)...)...);
}


__agency_exec_check_disable__
template<typename F, typename Tuple, size_t... I>
__AGENCY_ANNOTATION
auto apply_impl(F&& f, Tuple&& t, index_sequence<I...>)
  -> decltype(
       std::forward<F>(f)(
         agency::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return std::forward<F>(f)(
    agency::get<I>(std::forward<Tuple>(t))...
  );
}


} // end detail


template<class... Tuples>
__AGENCY_ANNOTATION
detail::tuple_cat_result_t<Tuples&&...> tuple_cat(Tuples&&... tuples)
{
  // compute the size of the tuple we will return
  static const std::size_t size = detail::tuple_cat_size<typename std::decay<Tuples>::type...>::value;

  // make an index sequence of that size and call the implementation
  return detail::tuple_cat_impl(detail::make_index_sequence<size>(), std::forward<Tuples>(tuples)...);
}


template<typename F, typename Tuple>
__AGENCY_ANNOTATION
auto apply(F&& f, Tuple&& t)
  -> decltype(
       detail::apply_impl(
         std::forward<F>(f),
         std::forward<Tuple>(t),
         detail::make_index_sequence<std::tuple_size<detail::decay_t<Tuple>>::value>()
       )
     )
{
  using Indices = detail::make_index_sequence<std::tuple_size<detail::decay_t<Tuple>>::value>;
  return detail::apply_impl(
    std::forward<F>(f),
    std::forward<Tuple>(t),
    Indices()
  );
}


namespace detail
{


template<class T>
struct invoke_constructor
{
  template<class... Args>
  __AGENCY_ANNOTATION
  T operator()(Args&&... args) const
  {
    return T(std::forward<Args>(args)...);
  }
};


} // end detail


template<class T, class Tuple>
__AGENCY_ANNOTATION
T make_from_tuple(Tuple&& t)
{
  return agency::apply(detail::invoke_constructor<T>{}, std::forward<Tuple>(t));
}


} // end agency


// specialize Tuple-related functionality in namespace std
namespace std
{


template<size_t i, class... Types>
class tuple_element<i, agency::tuple<Types...>>
{
  public:
    using type = typename std::tuple_element<i, agency::detail::tuple_base<agency::detail::make_index_sequence<sizeof...(Types)>, Types...>>::type;
};


template<class... Types>
class tuple_size<agency::tuple<Types...>>
  : public std::tuple_size<agency::detail::tuple_base<agency::detail::make_index_sequence<sizeof...(Types)>, Types...>>
{};


} // end std

