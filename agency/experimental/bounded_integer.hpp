#pragma once

#include <agency/detail/config.hpp>
#include <type_traits>
#include <cstdint>
#include <cstddef>
#include <limits>

namespace agency
{
namespace experimental
{


// a bounded_integer represents an integer known at compile time to be no greater than a given bound
// if arithmetic operations would cause overflow beyond the bound, the value of the resulting bounded_integer is undefined
template<class Integer, Integer bound>
class bounded_integer
{
  public:
    // XXX in principle, this class template could probably work for all arithmetic types
    static_assert(std::is_integral<Integer>::value, "Integer must be an integer type.");

    using value_type = Integer;

    static const value_type static_bound = bound;

    constexpr bounded_integer() = default;

    constexpr bounded_integer(const bounded_integer&) = default;

    // if number > bound, the value of the bounded_integer is undefined
    template<class Number,
             class = typename std::enable_if<
               std::is_constructible<value_type, Number>::value
             >::type>
    __AGENCY_ANNOTATION
    constexpr bounded_integer(const Number& number)
      : value_(number)
    {}

    // access the stored value
    __AGENCY_ANNOTATION
    constexpr const value_type& value() const
    {
      return value_;
    }

    // allow conversions to the value_type
    __AGENCY_ANNOTATION
    constexpr operator const value_type& () const
    {
      return value();
    }

    // operator members follow
    bounded_integer& operator=(const bounded_integer&) = default;

    // assign
    __AGENCY_ANNOTATION
    bounded_integer& operator=(const value_type& other)
    {
      value_ = other;
      return *this;
    }

    // pre increment
    __AGENCY_ANNOTATION
    bounded_integer& operator++()
    {
      ++value_;
      return *this;
    }

    // post-increment
    __AGENCY_ANNOTATION
    bounded_integer operator++(int)
    {
      bounded_integer result = *this;
      value_++;
      return result;
    }

    // plus-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator+=(const value_type& rhs)
    {
      value_ += rhs;
      return *this;
    }

    // minus-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator-=(const value_type& rhs)
    {
      value_ -= rhs;
      return *this;
    }

    // multiply-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator*=(const value_type& rhs)
    {
      value_ *= rhs;
      return *this;
    }

    // divide-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator/=(const value_type& rhs)
    {
      value_ /= rhs;
      return *this;
    }

    // modulus-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator%=(const value_type& rhs)
    {
      value_ %= rhs;
      return *this;
    }

    // left shift-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator<<=(const value_type& rhs)
    {
      value_ <<= rhs;
      return *this;
    }

    // right shift-assign
    __AGENCY_ANNOTATION
    bounded_integer& operator>>=(const value_type& rhs)
    {
      value_ >>= rhs;
      return *this;
    }

    // unary negate
    __AGENCY_ANNOTATION
    constexpr bool operator!() const
    {
      return !value();
    }

    // conversion to bool
    __AGENCY_ANNOTATION
    explicit constexpr operator bool() const
    {
      return value();
    }

  private:
    // XXX in principle, since we know the largest value this type can store,
    //     we could potentially implement the storage for the value with a
    //     narrower type
    using storage_type = value_type;
    storage_type value_;
};


// operators follow
// * the reason these are defined with trailing return type is two-fold:
//   1. to allow SFINAE to remove them from the overload set for operations on types which do not make sense
//   2. to deduce the result of arithmetic operations on mixed types
// * the reason there are three overloads for each operator is
//   1. to allow the bounded_integer to appear on either side of an expression
//   2. to disambiguate operations in which a bounded_integer appears on both sides of an expression


// plus
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator+(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() + rhs)
{
  return lhs.value() + rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator+(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs + rhs.value())
{
  return lhs + rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator+(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs + rhs.value())
{
  return lhs.value() + rhs.value();
}


// minus
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator-(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() - rhs)
{
  return lhs.value() - rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator-(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs - rhs.value())
{
  return lhs - rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator-(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs - rhs.value())
{
  return lhs.value() - rhs.value();
}


// multiply
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator*(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() * rhs)
{
  return lhs.value() * rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator*(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs * rhs.value())
{
  return lhs * rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator*(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs * rhs.value())
{
  return lhs.value() * rhs.value();
}


// divide
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator/(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() / rhs)
{
  return lhs.value() / rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator/(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs / rhs.value())
{
  return lhs / rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator/(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs / rhs.value())
{
  return lhs.value() / rhs.value();
}


// modulus
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator%(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() % rhs)
{
  return lhs.value() % rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator%(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs % rhs.value())
{
  return lhs % rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator%(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs % rhs.value())
{
  return lhs.value() % rhs.value();
}


// left shift
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator<<(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() << rhs)
{
  return lhs.value() << rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<<(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs << rhs.value())
{
  return lhs << rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<<(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs << rhs.value())
{
  return lhs.value() << rhs.value();
}


// right shift
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator>>(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() >> rhs)
{
  return lhs.value() >> rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>>(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs >> rhs.value())
{
  return lhs >> rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>>(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs >> rhs.value())
{
  return lhs.value() >> rhs.value();
}


// equal
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator==(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() == rhs)
{
  return lhs.value() == rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator==(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs == rhs.value())
{
  return lhs == rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator==(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs == rhs.value())
{
  return lhs.value() == rhs.value();
}


// not equal
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator!=(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() != rhs)
{
  return lhs.value() != rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator!=(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs != rhs.value())
{
  return lhs != rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator!=(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs != rhs.value())
{
  return lhs.value() != rhs.value();
}


// less
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator<(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() < rhs)
{
  return lhs.value() < rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs < rhs.value())
{
  return lhs < rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs < rhs.value())
{
  return lhs.value() < rhs.value();
}


// greater
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator>(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() > rhs)
{
  return lhs.value() > rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs > rhs.value())
{
  return lhs > rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs > rhs.value())
{
  return lhs.value() > rhs.value();
}


// less equal
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator<=(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() <= rhs)
{
  return lhs.value() <= rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<=(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs <= rhs.value())
{
  return lhs <= rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator<=(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs <= rhs.value())
{
  return lhs.value() <= rhs.value();
}


// greater equal
template<class Integer, Integer bound, class Number>
__AGENCY_ANNOTATION
constexpr auto operator>=(const bounded_integer<Integer,bound>& lhs, const Number& rhs) ->
  decltype(lhs.value() >= rhs)
{
  return lhs.value() >= rhs;
}

template<class Number, class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>=(const Number& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs >= rhs.value())
{
  return lhs >= rhs.value();
}

template<class Integer, Integer bound>
__AGENCY_ANNOTATION
constexpr auto operator>=(const bounded_integer<Integer,bound>& lhs, const bounded_integer<Integer,bound>& rhs) ->
  decltype(lhs >= rhs.value())
{
  return lhs.value() >= rhs.value();
}


// define some aliases for common integer types
template<int bound>
using bounded_int = bounded_integer<int, bound>;

template<unsigned int bound>
using bounded_uint = bounded_integer<unsigned int, bound>;

template<short bound>
using bounded_short = bounded_integer<short, bound>;

template<unsigned short bound>
using bounded_ushort = bounded_integer<unsigned short, bound>;


template<std::int8_t bound>
using bounded_int8_t = bounded_integer<std::int8_t, bound>;

template<std::int16_t bound>
using bounded_int16_t = bounded_integer<std::int16_t, bound>;

template<std::int32_t bound>
using bounded_int32_t = bounded_integer<std::int32_t, bound>;

template<std::int64_t bound>
using bounded_int64_t = bounded_integer<std::int64_t, bound>;


template<std::uint8_t bound>
using bounded_uint8_t = bounded_integer<std::uint8_t, bound>;

template<std::uint16_t bound>
using bounded_uint16_t = bounded_integer<std::uint16_t, bound>;

template<std::uint32_t bound>
using bounded_uint32_t = bounded_integer<std::uint32_t, bound>;

template<std::uint64_t bound>
using bounded_uint64_t = bounded_integer<std::uint64_t, bound>;


template<std::size_t bound>
using bounded_size_t = bounded_integer<std::size_t, bound>;


} // end experimental
} // end agency


// specialize std::numeric_limits for bounded_integer
namespace std
{


template<class Integer, Integer bound>
class numeric_limits<agency::experimental::bounded_integer<Integer,bound>>
  : public numeric_limits<Integer> // inherit the functionality of numeric_limits<Integer>
{
  public:
    // we can provide a specialization of max() since we know the bound
    __AGENCY_ANNOTATION
    static constexpr Integer max()
    {
      return bound;
    }
};


}

