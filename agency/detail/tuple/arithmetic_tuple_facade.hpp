#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <tuple>
#include <type_traits>
#include <iostream>

namespace agency
{
namespace detail
{

template<typename Derived>
class arithmetic_tuple_facade;


template<class T>
struct is_arithmetic_tuple : std::is_base_of<arithmetic_tuple_facade<T>, T> {};


template<typename Derived>
  class arithmetic_tuple_facade
{
  private:
    struct assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs = rhs;
      }
    };

    struct plus_assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs += rhs;
      }
    };

    struct minus_assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs -= rhs;
      }
    };

    struct multiplies_assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs *= rhs;
      }
    };

    template<typename T>
    struct assign_constant
    {
      T c;

      template<typename U>
      __AGENCY_ANNOTATION
      U& operator()(U& x) const
      {
        return x = c;
      }
    };

    template<typename T>
    struct multiplies_assign_constant
    {
      T c;

      template<typename U>
      __AGENCY_ANNOTATION
      U& operator()(U& x) const
      {
        return x *= c;
      }
    };

    struct divides_assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs /= rhs;
      }
    };

    template<typename T>
    struct divides_assign_constant
    {
      T c;

      template<typename U>
      __AGENCY_ANNOTATION
      U& operator()(U& x) const
      {
        return x /= c;
      }
    };

    struct modulus_assign
    {
      template<typename T1, typename T2>
      __AGENCY_ANNOTATION
      T1& operator()(T1 &lhs, const T2& rhs) const
      {
        return lhs %= rhs;
      }
    };

    struct make_derived
    {
      template<class... Args>
      __AGENCY_ANNOTATION
      Derived operator()(Args&&... args) const
      {
        return Derived{std::forward<Args>(args)...};
      }
    };

    __AGENCY_ANNOTATION Derived& derived()
    {
      return static_cast<Derived&>(*this);
    }

    __AGENCY_ANNOTATION const Derived& derived() const
    {
      return static_cast<const Derived&>(*this);
    }

  protected:
    template<class Arithmetic>
    __AGENCY_ANNOTATION
    typename std::enable_if<
      std::is_arithmetic<Arithmetic>::value
    >::type
      fill(const Arithmetic& val)
    {
      return __tu::tuple_for_each(assign_constant<Arithmetic>{val}, derived());
    }

    template<class ArithmeticTuple,
             class = typename std::enable_if<
               std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
             >::type>
    __AGENCY_ANNOTATION
    void copy(const ArithmeticTuple& src)
    {
      return __tu::tuple_for_each(assign{}, derived(), src);
    }

  public:

  // fused op-assignment
  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION Derived& operator*=(const ArithmeticTuple& rhs)
  {
    __tu::tuple_for_each(multiplies_assign{}, derived(), rhs);
    return derived();
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION Derived& operator/=(const ArithmeticTuple& rhs)
  {
    __tu::tuple_for_each(divides_assign{}, derived(), rhs);
    return derived();
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION Derived& operator%=(const ArithmeticTuple& rhs)
  {
    __tu::tuple_for_each(modulus_assign{}, derived(), rhs);
    return derived();
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION Derived& operator+=(const ArithmeticTuple& rhs)
  {
    __tu::tuple_for_each(plus_assign{}, derived(), rhs);
    return derived();
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION Derived& operator-=(const ArithmeticTuple& rhs)
  {
    __tu::tuple_for_each(minus_assign{}, derived(), rhs);
    return derived();
  }


  // multiply by scalar
  template<class Arithmetic>
  __AGENCY_ANNOTATION
  typename std::enable_if<
    std::is_arithmetic<Arithmetic>::value,
    Derived&
  >::type
    operator*=(const Arithmetic& rhs)
  {
    __tu::tuple_for_each(multiplies_assign_constant<Arithmetic>(rhs), derived());
    return derived();
  }

  // divide by scalar
  template<class Arithmetic>
  __AGENCY_ANNOTATION
  typename std::enable_if<
    std::is_arithmetic<Arithmetic>::value,
    Derived&
  >::type
    operator/=(const Arithmetic& rhs)
  {
    __tu::tuple_for_each(divides_assign_constant<Arithmetic>(rhs), derived());
    return derived();
  }


  // ops
  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  Derived operator+(const ArithmeticTuple& rhs) const
  {
    Derived result = derived();
    static_cast<arithmetic_tuple_facade&>(result) += rhs;
    return result;
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  Derived operator-(const ArithmeticTuple& rhs) const
  {
    Derived result = derived();
    static_cast<arithmetic_tuple_facade&>(result) -= rhs;
    return result;
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  Derived operator*(const ArithmeticTuple& rhs) const
  {
    Derived result = derived();
    static_cast<arithmetic_tuple_facade&>(result) *= rhs;
    return result;
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  Derived operator/(const ArithmeticTuple& rhs) const
  {
    Derived result = derived();
    static_cast<arithmetic_tuple_facade&>(result) /= rhs;
    return result;
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             std::tuple_size<Derived>::value == std::tuple_size<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  Derived operator%(const ArithmeticTuple& rhs) const
  {
    Derived result = derived();
    static_cast<arithmetic_tuple_facade&>(result) %= rhs;
    return result;
  }


  // equality
  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator==(const ArithmeticTuple& rhs) const
  {
    return __tu::tuple_equal(derived(), rhs);
  }


  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator!=(const ArithmeticTuple& rhs) const
  {
    return !operator==(rhs);
  }


  // relational ops

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator<(const ArithmeticTuple& rhs) const
  {
    return __tu::tuple_lexicographical_compare(derived(), rhs);
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator>(const ArithmeticTuple& rhs) const
  {
    return __tu::tuple_lexicographical_compare(rhs, derived());
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator<=(const ArithmeticTuple& rhs) const
  {
    return !operator>(rhs);
  }

  template<class ArithmeticTuple,
           class = typename std::enable_if<
             is_arithmetic_tuple<ArithmeticTuple>::value
           >::type>
  __AGENCY_ANNOTATION
  bool operator>=(const ArithmeticTuple& rhs) const
  {
    return !operator<(rhs);
  }

  friend std::ostream& operator<<(std::ostream& os, const arithmetic_tuple_facade& t)
  {
    os << "{";
    __tu::tuple_print(t.derived(), os, ", ");
    os << "}";

    return os;
  }
};


} // end detail
} // end agency

