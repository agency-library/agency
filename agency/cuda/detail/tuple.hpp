#pragma once

#include <thrust/tuple.h>
#include <type_traits>
#include <agency/detail/integer_sequence.hpp>
#include <cstddef>
#include <tuple>

namespace agency
{
namespace cuda
{
namespace detail
{
namespace tuple_detail
{

// XXX WAR nvbug 1527140
//     unpack template parameter packs into thrust::tuple manually
template<class... T>
struct tuple_war_1527140;

template<>
struct tuple_war_1527140<>
{
  using type = thrust::tuple<>;
};

template<class T1>
struct tuple_war_1527140<T1>
{
  using type = thrust::tuple<T1>;
};

template<class T1, class T2>
struct tuple_war_1527140<T1,T2>
{
  using type = thrust::tuple<T1,T2>;
};

template<class T1, class T2, class T3>
struct tuple_war_1527140<T1,T2,T3>
{
  using type = thrust::tuple<T1,T2,T3>;
};

template<class T1, class T2, class T3, class T4>
struct tuple_war_1527140<T1,T2,T3,T4>
{
  using type = thrust::tuple<T1,T2,T3,T4>;
};

template<class T1, class T2, class T3, class T4, class T5>
struct tuple_war_1527140<T1,T2,T3,T4,T5>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_war_1527140<T1,T2,T3,T4,T5,T6>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5,T6>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_war_1527140<T1,T2,T3,T4,T5,T6,T7>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5,T6,T7>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct tuple_war_1527140<T1,T2,T3,T4,T5,T6,T7,T8>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
struct tuple_war_1527140<T1,T2,T3,T4,T5,T6,T7,T8,T9>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>;
};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
struct tuple_war_1527140<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>
{
  using type = thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>;
};

} // end tuple_detail


template<class... T>
using tuple = typename tuple_detail::tuple_war_1527140<T...>::type;


// XXX replace this with the variadic forward_as_tuple() when thrust::tuple's constructor can receive && references
inline __host__ __device__
tuple<> forward_as_tuple()
{
  return tuple<>();
}


template<class T>
__host__ __device__
tuple<T&> forward_as_tuple(T& arg)
{
  return tuple<T&>(arg);
}


template<class T>
__host__ __device__
tuple<const T&> forward_as_tuple(const T& arg)
{
  return tuple<const T&>(arg);
}


template<class T1, class T2>
__host__ __device__
tuple<T1&,T2&> forward_as_tuple(T1& arg1, T2& arg2)
{
  return tuple<T1&,T2&>(arg1, arg2);
}


template<class T1, class T2>
__host__ __device__
tuple<T1&,const T2&> forward_as_tuple(T1& arg1, const T2& arg2)
{
  return tuple<T1&,const T2&>(arg1, arg2);
}


template<class T1, class T2>
__host__ __device__
tuple<const T1&,T2&> forward_as_tuple(const T1& arg1, T2& arg2)
{
  return tuple<const T1&,T2&>(arg1, arg2);
}


template<class T1, class T2>
__host__ __device__
tuple<const T1&,const T2&> forward_as_tuple(const T1& arg1, const T2& arg2)
{
  return tuple<const T1&,const T2&>(arg1, arg2);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<T1&,T2&,T3&> forward_as_tuple(T1& arg1, T2& arg2, T3& arg3)
{
  return tuple<T1&,T2&,T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<T1&,T2&,const T3&> forward_as_tuple(T1& arg1, T2& arg2, const T3& arg3)
{
  return tuple<T1&,T2&,const T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<T1&,const T2&,T3&> forward_as_tuple(T1& arg1, const T2& arg2, T3& arg3)
{
  return tuple<T1&,const T2&, T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<T1&,const T2&,const T3&> forward_as_tuple(T1& arg1, const T2& arg2, const T3& arg3)
{
  return tuple<T1&,const T2&,const T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<const T1&,T2&,T3&> forward_as_tuple(const T1& arg1, T2& arg2, T3& arg3)
{
  return tuple<const T1&,T2&,T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<const T1&,T2&,const T3&> forward_as_tuple(const T1& arg1, T2& arg2, const T3& arg3)
{
  return tuple<const T1&,T2&,const T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<const T1&,const T2&,T3&> forward_as_tuple(const T1& arg1, const T2& arg2, T3& arg3)
{
  return tuple<const T1&,const T2&, T3&>(arg1, arg2, arg3);
}


template<class T1, class T2, class T3>
__host__ __device__
tuple<const T1&,const T2&,const T3&> forward_as_tuple(const T1& arg1, const T2& arg2, const T3& arg3)
{
  return tuple<const T1&,const T2&,const T3&>(arg1, arg2, arg3);
}



template<class, class> struct tuple_of_references_impl;


template<class Tuple, size_t... I>
struct tuple_of_references_impl<Tuple,agency::detail::index_sequence<I...>>
{
  using type = tuple<
    typename std::tuple_element<I,Tuple>::type...
  >;
};


template<class Tuple>
using tuple_of_references_t =
  typename tuple_of_references_impl<
    Tuple,
    agency::detail::make_index_sequence<std::tuple_size<Tuple>::value>
  >::type;


} // end detail
} // end cuda
} // end agency


namespace std
{


// implement the std tuple interface for thrust::tuple
// XXX we'd specialize these for cuda::detail::tuple, but we can't specialize on template using


template<class Type1, class... Types>
struct tuple_size<thrust::tuple<Type1,Types...>> : thrust::tuple_size<thrust::tuple<Type1,Types...>> {};


template<size_t i, class Type1, class... Types>
struct tuple_element<i,thrust::tuple<Type1,Types...>> : thrust::tuple_element<i,thrust::tuple<Type1,Types...>> {};


} // end std


namespace __tu
{


// tuple_traits specialization

template<class Type1, class... Types>
struct tuple_traits<thrust::tuple<Type1,Types...>>
{
  using tuple_type = thrust::tuple<Type1,Types...>;

  static const size_t size = thrust::tuple_size<tuple_type>::value;

  template<size_t i>
  using element_type = typename thrust::tuple_element<i,tuple_type>::type;

  template<size_t i>
  __AGENCY_ANNOTATION
  static element_type<i>& get(tuple_type& t)
  {
    return thrust::get<i>(t);
  } // end get()

  template<size_t i>
  __AGENCY_ANNOTATION
  static const element_type<i>& get(const tuple_type& t)
  {
    return thrust::get<i>(t);
  } // end get()
}; // end tuple_traits


} // end __tu



