#pragma once

// have to declare std::get for small_vector_facade
// before deriving from arithmetic_tuple_facade
// because std::get can't be looked up via ADL
namespace agency
{
namespace detail
{


template<typename Derived, typename T, std::size_t Rank>
class small_vector_facade;


} // end detail
} // end agency


namespace std
{


template<size_t I, class Derived, class T, size_t Rank>
T& get(agency::detail::small_vector_facade<Derived,T,Rank>& x);


template<size_t I, class Derived, class T, size_t Rank>
const T& get(const agency::detail::small_vector_facade<Derived,T,Rank>& x);


}


#include <type_traits>
#include <algorithm>
#include <agency/detail/operator_traits.hpp>
#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <iostream>
#include <array>
#include <utility>

namespace agency
{
namespace detail
{


template<bool Value1, bool... Values>
struct all_of
  : std::integral_constant<bool, Value1 && all_of<Values...>::value>
{};


template<bool Value1> struct all_of<Value1>
  : std::integral_constant<bool, Value1>
{};


template<typename Derived, typename T, std::size_t Rank>
  class small_vector_facade : public agency::detail::arithmetic_tuple_facade<Derived>
{
  static_assert(agency::detail::has_arithmetic_operators<T>::value, "T must have arithmetic operators.");
  static_assert(Rank > 0, "Rank must be greater than 0.");

  public:
    using value_type                = T;
    using reference                 = value_type&;
    using const_reference           = const value_type&;
    using size_type                 = size_t;
    using pointer                   = value_type*;
    using const_pointer             = const value_type*;
    using iterator                  = pointer;
    using const_iterator            = const_pointer;
    static constexpr size_type rank = Rank;

    constexpr size_type size() const
    {
      return rank;
    }

    reference front()
    {
      return *begin();
    }

    const_reference front() const
    {
      return *begin();
    }

    reference back()
    {
      return *(end() - 1);
    }

    const_reference back() const
    {
      return *(end() - 1);
    }

    pointer data()
    {
      return derived();
    }

    const_pointer data() const
    {
      return derived();
    }

    iterator begin()
    {
      return data();
    }

    iterator end()
    {
      return data() + size();
    }

    const_iterator begin() const
    {
      return data();
    }

    const_iterator end() const
    {
      return data() + size();
    }

    const_iterator cbegin() const
    {
      return begin();
    }

    const_iterator cend() const
    {
      return end();
    }

    bool operator==(const small_vector_facade& rhs) const
    {
      return std::equal(begin(), end(), rhs.begin());
    }


    bool operator!=(const small_vector_facade& rhs) const
    {
      return !(*this == rhs);
    }


    reference operator[](size_type i)
    {
      return begin()[i];
    }


    const_reference operator[](size_type i) const
    {
      return begin()[i];
    }


  private:
    inline Derived &derived()
    {
      return static_cast<Derived&>(*this);
    }

    inline const Derived &derived() const
    {
      return static_cast<const Derived&>(*this);
    }
};


} // end detail
} // end agency


namespace std
{


template<size_t I, class Derived, class T, size_t Rank>
T& get(agency::detail::small_vector_facade<Derived,T,Rank>& x)
{
  static_assert(I < Rank, "I must be less than Rank.");
  return x[I];
}


template<size_t I, class Derived, class T, size_t Rank>
const T& get(const agency::detail::small_vector_facade<Derived,T,Rank>& x)
{
  static_assert(I < Rank, "I must be less than Rank.");
  return x[I];
}


} // end std


namespace agency
{
namespace detail
{


template<typename Derived, typename Base, typename T, std::size_t Rank>
  class small_vector_adaptor : public small_vector_facade<Derived, T, Rank>
{
  static_assert(agency::detail::has_arithmetic_operators<T>::value, "T must have arithmetic operators.");
  static_assert(Rank > 0, "Rank must be greater than 0.");

  using super_t = small_vector_facade<Derived,T,Rank>;

  public:
    using typename super_t::value_type;
    using typename super_t::reference;
    using typename super_t::size_type;
    using typename super_t::pointer;
    using typename super_t::const_pointer;


    small_vector_adaptor()
      : small_vector_adaptor(value_type{})
    {
      for(size_type i = 0; i != super_t::rank; ++i)
      {
        base()[i] = value_type{};
      }
    }


    small_vector_adaptor(const small_vector_adaptor &other)
    {
      for(size_type i = 0; i != super_t::rank; ++i)
      {
        base()[i] = other[i];
      }
    }


    template<class... OtherT,
             typename = typename std::enable_if<
               all_of<
                 std::is_convertible<OtherT,value_type>::value...
               >::value &&
               sizeof...(OtherT) == Rank
             >::type>
    small_vector_adaptor(OtherT... args)
      : small_vector_adaptor{static_cast<value_type>(args)...}
    {}


    // XXX need to only enable this if the initializer_list is the right size
    //     but we can't do that until c++14
    template<class OtherT,
             typename = typename std::enable_if<
               std::is_convertible<OtherT,value_type>::value
             >::type>
    small_vector_adaptor(std::initializer_list<OtherT> l)
    {
      std::copy(l.begin(), l.end(), super_t::begin());
    }


    // XXX should fully parameterize this
    template<class OtherDerived, class OtherT,
             class = typename std::enable_if<
               std::is_convertible<OtherT,value_type>::value
             >::type>
    small_vector_adaptor(const small_vector_facade<OtherDerived,OtherT,Rank>& other)
    {
      for(size_type i = 0; i < super_t::size(); ++i)
      {
        super_t::operator[](i) = other[i];
      }
    }


    // fills the vector with a constant value
    template<class OtherT,
             class = typename std::enable_if<
               (Rank > 1) &&
               std::is_convertible<OtherT,value_type>::value
             >::type
             >
    small_vector_adaptor(OtherT val)
    {
      std::fill(super_t::begin(),super_t::end(),val);
    }


    operator pointer ()
    {
      return base().data();
    }


    operator const_pointer () const
    {
      return base().data();
    }


  protected:
    Base& base()
    {
      return base_;
    }

    const Base& base() const
    {
      return base_;
    }

  private:
    Base base_;
};


} // end detail
} // end agency

