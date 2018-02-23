#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/tuple/arithmetic_tuple_facade.hpp>
#include <agency/detail/operator_traits.hpp>
#include <agency/container/array.hpp>
#include <agency/coordinate/detail/named_array.hpp>

#include <type_traits>
#include <initializer_list>
#include <cassert>

namespace agency
{
namespace detail
{


// in general, point's base class is array<T,Rank> but low-rank points get named elements
template<class T, size_t Rank>
struct point_base
{
  using type = array<T,Rank>;
};

template<class T>
struct point_base<T,1>
{
  using type = named_array<T,1>;
};

template<class T>
struct point_base<T,2>
{
  using type = named_array<T,2>;
};

template<class T>
struct point_base<T,3>
{
  using type = named_array<T,3>;
};

template<class T>
struct point_base<T,4>
{
  using type = named_array<T,4>;
};

template<class T, size_t Rank>
using point_base_t = typename point_base<T,Rank>::type;


} // end detail


// T is any type with operators +, +=, -, -=, *, *=,  /, /=, <
template<class T, size_t Rank>
class point : public agency::detail::point_base_t<T,Rank>,
              public agency::detail::arithmetic_tuple_facade<point<T,Rank>>
{
  static_assert(agency::detail::has_arithmetic_operators<T>::value, "T must have arithmetic operators.");

  using super_t = detail::point_base_t<T,Rank>;

  public:
    using typename super_t::value_type;
    using typename super_t::reference;
    using typename super_t::size_type;
    using typename super_t::pointer;
    using typename super_t::const_pointer;


    point() = default;


    point(const point &) = default;


    template<class... OtherT,
             __AGENCY_REQUIRES(
               detail::conjunction<
                 std::is_convertible<OtherT,value_type>...
               >::value &&
               sizeof...(OtherT) == Rank
             )>
    __AGENCY_ANNOTATION
    point(OtherT... args)
      : super_t{{static_cast<value_type>(args)...}}
    {
    }


    // this constructor is included to allow us to pass curly-braced lists through
    // interfaces which eventually get unpacked into points
    // for example, in expressions like this:
    //
    //     auto policy = agency::par2d({0,0}, {5,5});
    //
    // XXX trying to initialize a point from an initializer_list of the wrong size
    //     should be a compile-time error
    //     the problem is that l.size() can't always be used in static_assert
    template<class OtherT,
             __AGENCY_REQUIRES(
               std::is_convertible<OtherT,value_type>::value
             )>
    __AGENCY_ANNOTATION
    point(std::initializer_list<OtherT> l)
    {
      // l.size() needs to equal Rank
      assert(l.size() == Rank);

      auto src = l.begin();
      for(auto dst = super_t::begin(); dst != super_t::end(); ++src, ++dst)
      {
        *dst = *src;
      }
    }


    // XXX should fully parameterize this
    template<class OtherT,
             __AGENCY_REQUIRES(
               std::is_convertible<OtherT,value_type>::value
             )>
    __AGENCY_ANNOTATION
    point(const point<OtherT,Rank>& other)
    {
      detail::arithmetic_tuple_facade<point>::copy(other);
    }


    // fills the point with a constant value
    template<class OtherT,
             __AGENCY_REQUIRES(
               (Rank > 1) &&
               std::is_convertible<OtherT,value_type>::value
             )>
    __AGENCY_ANNOTATION
    explicit point(OtherT val)
    {
      detail::arithmetic_tuple_facade<point>::fill(val);
    }


    // XXX this should be eliminated
    __AGENCY_ANNOTATION
    operator pointer ()
    {
      return super_t::data();
    }


    // XXX this should be eliminated
    __AGENCY_ANNOTATION
    operator const_pointer () const
    {
      return super_t::data();
    }
};


template<size_t i, class T, size_t Rank>
__AGENCY_ANNOTATION
T& get(point<T,Rank>& p)
{
  return p[i];
}


template<size_t i, class T, size_t Rank>
__AGENCY_ANNOTATION
const T& get(const point<T,Rank>& p)
{
  return p[i];
}


template<size_t i, class T, size_t Rank>
__AGENCY_ANNOTATION
T&& get(point<T,Rank>&& p)
{
  return std::move(agency::get<i>(p));
}


// scalar multiply
// XXX fix return type -- it should be point<common_type,Rank>
template<class T1, class T2, size_t Rank>
__AGENCY_ANNOTATION
typename std::enable_if<
  (std::is_arithmetic<T1>::value && agency::detail::has_operator_multiplies<T1,T2>::value),
  point<T2,Rank>
>::type
  operator*(T1 val, const point<T2,Rank>& p)
{
  using result_type = point<T2, Rank>;

  return result_type(val) * p;
}


using int0  = point<int,0>;
using int1  = point<int,1>;
using int2  = point<int,2>;
using int3  = point<int,3>;
using int4  = point<int,4>;
using int5  = point<int,5>;
using int6  = point<int,6>;
using int7  = point<int,7>;
using int8  = point<int,8>;
using int9  = point<int,9>;
using int10 = point<int,10>;


using uint0  = point<unsigned int,0>;
using uint1  = point<unsigned int,1>;
using uint2  = point<unsigned int,2>;
using uint3  = point<unsigned int,3>;
using uint4  = point<unsigned int,4>;
using uint5  = point<unsigned int,5>;
using uint6  = point<unsigned int,6>;
using uint7  = point<unsigned int,7>;
using uint8  = point<unsigned int,8>;
using uint9  = point<unsigned int,9>;
using uint10 = point<unsigned int,10>;


using size0  = point<size_t,0>;
using size1  = point<size_t,1>;
using size2  = point<size_t,2>;
using size3  = point<size_t,3>;
using size4  = point<size_t,4>;
using size5  = point<size_t,5>;
using size6  = point<size_t,6>;
using size7  = point<size_t,7>;
using size8  = point<size_t,8>;
using size9  = point<size_t,9>;
using size10 = point<size_t,10>;


using float0  = point<float,0>;
using float1  = point<float,1>;
using float2  = point<float,2>;
using float3  = point<float,3>;
using float4  = point<float,4>;
using float5  = point<float,5>;
using float6  = point<float,6>;
using float7  = point<float,7>;
using float8  = point<float,8>;
using float9  = point<float,9>;
using float10 = point<float,10>;


using double0  = point<double,0>;
using double1  = point<double,1>;
using double2  = point<double,2>;
using double3  = point<double,3>;
using double4  = point<double,4>;
using double5  = point<double,5>;
using double6  = point<double,6>;
using double7  = point<double,7>;
using double8  = point<double,8>;
using double9  = point<double,9>;
using double10 = point<double,10>;


} // end agency


// specialize Tuple-like interface for agency::point
namespace std
{


template<class T, size_t Rank>
class tuple_size<agency::point<T,Rank>> : public std::integral_constant<std::size_t, Rank> {};


template<size_t I, class T, size_t Rank>
struct tuple_element<I,agency::point<T,Rank>>
{
  using type = T;
};


} // end std

