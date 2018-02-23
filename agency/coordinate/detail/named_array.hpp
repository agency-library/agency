#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/utility.hpp>
#include <cstddef>
#include <tuple>
#include <utility>


namespace agency
{
namespace detail
{


template<class T, size_t Index>
struct named_array_element;

template<class T>
struct named_array_element<T,0>
{
  named_array_element() = default;

  named_array_element(const named_array_element&) = default;

  __AGENCY_ANNOTATION
  explicit named_array_element(const T& value) : x(value) {}

  T x;
};

template<class T>
struct named_array_element<T,1>
{
  named_array_element() = default;

  named_array_element(const named_array_element&) = default;

  __AGENCY_ANNOTATION
  explicit named_array_element(const T& value) : y(value) {}

  T y;
};

template<class T>
struct named_array_element<T,2>
{
  named_array_element() = default;

  named_array_element(const named_array_element&) = default;

  __AGENCY_ANNOTATION
  explicit named_array_element(const T& value) : z(value) {}

  T z;
};

template<class T>
struct named_array_element<T,3>
{
  named_array_element() = default;

  named_array_element(const named_array_element&) = default;

  __AGENCY_ANNOTATION
  explicit named_array_element(const T& value) : w(value) {}

  T w;
};


template<class T, class Indices>
struct named_array_base;

template<class T, size_t... Indices>
struct named_array_base<T, index_sequence<Indices...>> : named_array_element<T,Indices>...
{
  named_array_base() = default;

  named_array_base(const named_array_base&) = default;

  __AGENCY_ANNOTATION
  named_array_base(std::initializer_list<T> values)
    : named_array_element<T,Indices>(values.begin()[Indices])...
  {}
};


// a named_array is an array where each element has its own name, e.g. [x, y, z, w]
template<class T, size_t N>
struct named_array : named_array_base<T,make_index_sequence<N>>
{
  static_assert(0 < N && N < 5, "named_array's size must be at least one and less than five.");

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

  named_array() = default;

  named_array(const named_array&) = default;

  __AGENCY_ANNOTATION
  named_array(std::initializer_list<T> values)
    : named_array_base<T,make_index_sequence<N>>(values)
  {}
  
  __AGENCY_ANNOTATION
  reference operator[](size_type pos)
  {
    return begin()[pos];
  }
  
  __AGENCY_ANNOTATION
  const_reference operator[](size_type pos) const
  {
    return begin()[pos];
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
    // return *rbegin();
    return operator[](N-1);
  }
  
  __AGENCY_ANNOTATION
  const_reference back() const
  {
    // return *rbegin();
    return operator[](N-1);
  }
  
  __AGENCY_ANNOTATION
  T* data()
  {
    return &this->x;
  }
  
  __AGENCY_ANNOTATION
  const T* data() const
  {
    return &this->x;
  }
  
  __AGENCY_ANNOTATION
  iterator begin()
  {
    return data();
  }
  
  __AGENCY_ANNOTATION
  const_iterator begin() const
  {
    return data();
  }
  
  __AGENCY_ANNOTATION
  const_iterator cbegin()
  {
    return begin();
  }
  
  __AGENCY_ANNOTATION
  const_iterator cbegin() const
  {
    return begin();
  }
  
  __AGENCY_ANNOTATION
  iterator end()
  {
    return data() + size();
  }
  
  __AGENCY_ANNOTATION
  const_iterator end() const
  {
    return data() + size();
  }
  
  __AGENCY_ANNOTATION
  const_iterator cend()
  {
    return end();
  }
  
  __AGENCY_ANNOTATION
  const_iterator cend() const
  {
    return end();
  }
  
  __AGENCY_ANNOTATION
  constexpr bool empty() const
  {
    return size() == 0;
  }
  
  __AGENCY_ANNOTATION
  constexpr size_type size() const
  {
    return N;
  }
  
  __AGENCY_ANNOTATION
  constexpr size_type max_size() const
  {
    return size();
  }
  
  __AGENCY_ANNOTATION
  void fill(const T& value)
  {
    for(auto& e : *this)
    {
      e = value;
    }
  }
  
  __AGENCY_ANNOTATION
  void swap(named_array& other)
  {
    for(size_type i = 0; i < size(); ++i)
    {
      agency::detail::adl_swap((*this)[i], other[i]);
    }
  }


  template<std::size_t I, __AGENCY_REQUIRES(I < N)>
  __AGENCY_ANNOTATION
  T& get() &
  {
    return operator[](I);
  }
  
  
  template<std::size_t I, __AGENCY_REQUIRES(I < N)>
  __AGENCY_ANNOTATION
  const T& get() const &
  {
    return operator[](I);
  }
  
  
  template<std::size_t I, __AGENCY_REQUIRES(I < N)>
  __AGENCY_ANNOTATION
  T&& get() &&
  {
    return std::move(operator[](I));
  }
};


} // end detail
} // end agency


// specialize tuple-related functionality for agency::detail::named_array
namespace std
{


template<class T, std::size_t N>
struct tuple_size<agency::detail::named_array<T,N>> : std::integral_constant<std::size_t, N> {};


template<std::size_t I, class T, std::size_t N>
struct tuple_element<I, agency::detail::named_array<T,N>>
{
  using type = T;
};


} // end std

