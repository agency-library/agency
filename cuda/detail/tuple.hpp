#pragma once

#include <thrust/tuple.h>
#include <type_traits>
#include <agency/detail/integer_sequence.hpp>
#include <cstddef>
#include <tuple>

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


namespace std
{


// implement the std tuple interface for thrust::tuple
// XXX we'd specialize these for cuda::detail::tuple, but we can't specialize on template using


template<class T1>
struct tuple_size<thrust::tuple<T1>> : std::integral_constant<size_t, 1> {};

template<class T1, class T2>
struct tuple_size<thrust::tuple<T1,T2>> : std::integral_constant<size_t, 2> {};

template<class T1, class T2, class T3>
struct tuple_size<thrust::tuple<T1,T2,T3>> : std::integral_constant<size_t, 3> {};

template<class T1, class T2, class T3, class T4>
struct tuple_size<thrust::tuple<T1,T2,T3,T4>> : std::integral_constant<size_t, 4> {};

template<class T1, class T2, class T3, class T4, class T5>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5>> : std::integral_constant<size_t, 5> {};

template<class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5,T6>> : std::integral_constant<size_t, 6> {};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5,T6,T7>> : std::integral_constant<size_t, 7> {};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>> : std::integral_constant<size_t, 8> {};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>> : std::integral_constant<size_t, 9> {};

template<class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
struct tuple_size<thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>> : std::integral_constant<size_t, 10> {};


template<size_t i, class T1>
struct tuple_element<i,thrust::tuple<T1>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1>>::type;
};

template<size_t i, class T1, class T2>
struct tuple_element<i,thrust::tuple<T1,T2>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2>>::type;
};

template<size_t i, class T1, class T2, class T3>
struct tuple_element<i,thrust::tuple<T1,T2,T3>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>>::type;
};

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
struct tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>>
{
  using type = typename thrust::tuple_element<i,thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>>::type;
};


template<size_t i, class T1>
__host__ __device__
auto get(thrust::tuple<T1>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1>
__host__ __device__
auto get(const thrust::tuple<T1>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2>
__host__ __device__
auto get(thrust::tuple<T1,T2>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2>
__host__ __device__
auto get(const thrust::tuple<T1,T2>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5,T6>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5,T6>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5,T6,T7>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5,T6,T7>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
__host__ __device__
auto get(thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}

template<size_t i, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9, class T10>
__host__ __device__
auto get(const thrust::tuple<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10>& t)
  -> decltype(thrust::get<i>(t))
{
  return thrust::get<i>(t);
}


} // end std

