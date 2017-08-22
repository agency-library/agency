#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/utility.hpp>
#include <cstddef>
#include <tuple>
#include <utility>


namespace agency
{


template<class T, std::size_t N>
struct array
{
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;
  using iterator = pointer;
  using const_iterator = const_pointer;

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
    return __elems_;
  }

  __AGENCY_ANNOTATION
  const T* data() const
  {
    return __elems_;
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
  void swap(array& other)
  {
    for(size_type i = 0; i < size(); ++i)
    {
      agency::detail::adl_swap((*this)[i], other[i]);
    }
  }

  value_type __elems_[N > 0 ? N : 1];
};


template<class T, std::size_t N>
__AGENCY_ANNOTATION
bool operator==(const array<T,N>& lhs,  const array<T,N>& rhs)
{
  for(std::size_t i = 0; i < N; ++i)
  {
    if(lhs[i] != rhs[i]) return false;
  }

  return true;
}


template<std::size_t I, class T, std::size_t N, __AGENCY_REQUIRES(I < N)>
__AGENCY_ANNOTATION
T& get(array<T,N>& a)
{
  return a[I];
}


template<std::size_t I, class T, std::size_t N, __AGENCY_REQUIRES(I < N)>
__AGENCY_ANNOTATION
const T& get(const array<T,N>& a)
{
  return a[I];
}


template<std::size_t I, class T, std::size_t N, __AGENCY_REQUIRES(I < N)>
__AGENCY_ANNOTATION
T&& get(array<T,N>&& a)
{
  return std::move(a[I]);
}


// XXX other relational operators here


template<class T, std::size_t N>
__AGENCY_ANNOTATION
void swap(array<T,N>& a, array<T,N>& b)
{
  a.swap(b);
}


// specialize range_cardinality for array
template<class Range>
struct range_cardinality;


template<class T, std::size_t N>
struct range_cardinality<array<T,N>> : std::integral_constant<size_t, N> {};


} // end agency


// specialize tuple-related functionality for agency::array
namespace std
{


template<class T, std::size_t N>
struct tuple_size<agency::array<T,N>> : std::integral_constant<std::size_t, N> {};


template<std::size_t I, class T, std::size_t N>
struct tuple_element<I, agency::array<T,N>>
{
  using type = T;
};


} // end std

