#pragma once

#include <agency/detail/config.hpp>

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
__AGENCY_ANNOTATION
T& get(agency::detail::small_vector_facade<Derived,T,Rank>& x);


template<size_t I, class Derived, class T, size_t Rank>
__AGENCY_ANNOTATION
const T& get(const agency::detail::small_vector_facade<Derived,T,Rank>& x);


}


#include <type_traits>
#include <agency/detail/operator_traits.hpp>
#include <agency/detail/arithmetic_tuple_facade.hpp>
#include <iostream>
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

    __AGENCY_ANNOTATION
    size_type size() const
    {
      return rank;
    }

    __AGENCY_ANNOTATION
    reference front()
    {
      return *begin();
    }

    __AGENCY_ANNOTATION
    const_reference front() const
    {
      return *begin();
    }

    __AGENCY_ANNOTATION
    reference back()
    {
      return *(end() - 1);
    }

    __AGENCY_ANNOTATION
    const_reference back() const
    {
      return *(end() - 1);
    }

    __AGENCY_ANNOTATION
    pointer data()
    {
      return derived();
    }

    __AGENCY_ANNOTATION
    const_pointer data() const
    {
      return derived();
    }

    __AGENCY_ANNOTATION
    iterator begin()
    {
      return data();
    }

    __AGENCY_ANNOTATION
    iterator end()
    {
      return data() + size();
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return data();
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return data() + size();
    }

    __AGENCY_ANNOTATION
    const_iterator cbegin() const
    {
      return begin();
    }

    __AGENCY_ANNOTATION
    const_iterator cend() const
    {
      return end();
    }


    __AGENCY_ANNOTATION
    bool operator==(const small_vector_facade& rhs) const
    {
      return std::equal(begin(), end(), rhs.begin());
    }


    __AGENCY_ANNOTATION
    bool operator!=(const small_vector_facade& rhs) const
    {
      return !(*this == rhs);
    }


    __AGENCY_ANNOTATION
    reference operator[](size_type i)
    {
      return begin()[i];
    }


    __AGENCY_ANNOTATION
    const_reference operator[](size_type i) const
    {
      return begin()[i];
    }


  private:
    __AGENCY_ANNOTATION
    inline Derived &derived()
    {
      return static_cast<Derived&>(*this);
    }

    __AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
T& get(agency::detail::small_vector_facade<Derived,T,Rank>& x)
{
  static_assert(I < Rank, "I must be less than Rank.");
  return x[I];
}


template<size_t I, class Derived, class T, size_t Rank>
__AGENCY_ANNOTATION
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


    __AGENCY_ANNOTATION
    small_vector_adaptor()
    {
      super_t::fill(value_type{});
    }


    __AGENCY_ANNOTATION
    small_vector_adaptor(const small_vector_adaptor &other)
      : base_(other.base_)
    {
    }


    template<class... OtherT,
             typename = typename std::enable_if<
               all_of<
                 std::is_convertible<OtherT,value_type>::value...
               >::value &&
               sizeof...(OtherT) == Rank
             >::type>
    __AGENCY_ANNOTATION
    small_vector_adaptor(OtherT... args)
      : base_{static_cast<value_type>(args)...}
    {
    }


    // XXX need to only enable this if the initializer_list is the right size
    //     but we can't do that until c++14
    template<class OtherT,
             typename = typename std::enable_if<
               std::is_convertible<OtherT,value_type>::value
             >::type>
    __AGENCY_ANNOTATION
    small_vector_adaptor(std::initializer_list<OtherT> l)
    {
      // XXX should try to use base_'s constructor with l instead of this for loop
      auto src = l.begin();
      for(auto dst = super_t::begin(); dst != super_t::end(); ++src, ++dst)
      {
        *dst = *src;
      }
    }


    // XXX should fully parameterize this
    template<class OtherDerived, class OtherT,
             class = typename std::enable_if<
               std::is_convertible<OtherT,value_type>::value
             >::type>
    __AGENCY_ANNOTATION
    small_vector_adaptor(const small_vector_facade<OtherDerived,OtherT,Rank>& other)
    {
      super_t::copy(other);
    }


    // fills the vector with a constant value
    template<class OtherT,
             class = typename std::enable_if<
               (Rank > 1) &&
               std::is_convertible<OtherT,value_type>::value
             >::type
             >
    __AGENCY_ANNOTATION
    small_vector_adaptor(OtherT val)
    {
      super_t::fill(val);
    }


    __AGENCY_ANNOTATION
    operator pointer ()
    {
      return base_.data();
    }


    __AGENCY_ANNOTATION
    operator const_pointer () const
    {
      return base_.data();
    }

  private:
    Base base_;
};


} // end detail
} // end agency

