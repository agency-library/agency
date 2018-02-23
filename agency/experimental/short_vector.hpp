#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/utility.hpp>
#include <agency/container/array.hpp>
#include <agency/experimental/optional.hpp>
#include <agency/experimental/bounded_integer.hpp>
#include <cstddef>
#include <utility>


namespace agency
{
namespace experimental
{
namespace detail
{
namespace short_vector_detail
{


// XXX eliminate this once we've integrated this functionality elsewhere within the library
template<size_t first, size_t last>
struct static_for_loop_impl
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&& f)
  {
    std::forward<Function>(f)(first);

    static_for_loop_impl<first+1,last>::invoke(std::forward<Function>(f));
  }
};


template<size_t first>
struct static_for_loop_impl<first,first>
{
  template<class Function>
  __AGENCY_ANNOTATION
  static void invoke(Function&&)
  {
  }
};


template<size_t n, class Function>
__AGENCY_ANNOTATION
void static_for_loop(Function&& f)
{
  static_for_loop_impl<0,n>::invoke(std::forward<Function>(f));
}


template<size_t first, size_t last>
struct bounded_index_impl
{
  template<class T, size_t N>
  __AGENCY_ANNOTATION
  static T& index(array<T,N>& a, int idx)
  {
    if(first == idx)
    {
      return a[first];
    }

    return bounded_index_impl<first+1,last>::index(a, idx);
  }

  template<class T, size_t N>
  __AGENCY_ANNOTATION
  static const T& index(const array<T,N>& a, int idx)
  {
    if(first == idx)
    {
      return a[first];
    }

    return bounded_index_impl<first+1,last>::index(a, idx);
  }
};


template<size_t first>
struct bounded_index_impl<first,first>
{
  template<class T, size_t N>
  __AGENCY_ANNOTATION
  static T& index(array<T,N>& a, int)
  {
    return a[0];
  }

  template<class T, size_t N>
  __AGENCY_ANNOTATION
  static const T& index(const array<T,N>& a, int)
  {
    return a[0];
  }
};


template<class T, size_t N>
__AGENCY_ANNOTATION
T& bounded_index(array<T,N>& a, int idx)
{
  return bounded_index_impl<0,N>::index(a, idx);
}

template<class T, size_t N>
__AGENCY_ANNOTATION
const T& bounded_index(const array<T,N>& a, int idx)
{
  return bounded_index_impl<0,N>::index(a, idx);
}


} // end short_vector_detail
} // end detail


template<class T, std::size_t N>
class short_vector
{
  private:
    template<class Function>
    __AGENCY_ANNOTATION
    void for_loop(Function&& f)
    {
      if(N == size())
      {
        detail::short_vector_detail::static_for_loop<N>(std::forward<Function>(f));
      }
      else
      {
        detail::short_vector_detail::static_for_loop<N>([&](std::size_t i)
        {
          if(i < size())
          {
            std::forward<Function>(f)(i);
          }
        });
      }
    }

  public:
    static const std::size_t static_max_size = N;

    using value_type = T;

    // encode the maximum size of this short_vector in its size_type
    using size_type = bounded_size_t<static_max_size>;

    using difference_type = std::ptrdiff_t;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using iterator = pointer;
    using const_iterator = const_pointer;

    __AGENCY_ANNOTATION
    short_vector()
      : size_(0)
    {
    }

    short_vector(const short_vector&) = default;

    short_vector(short_vector&&) = default;

    template<class Range>
    __AGENCY_ANNOTATION
    short_vector(Range&& other)
      : size_(other.size())
    {
      // copy construct each element with placement new
      for_loop([&](int i)
      {
        T& x = (*this)[i];
        ::new(&x) T(other[i]);
      });
    }

    __AGENCY_ANNOTATION
    reference operator[](size_type pos)
    {
      return array_[pos];
    }

    __AGENCY_ANNOTATION
    const_reference operator[](size_type pos) const
    {
      return array_[pos];
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
#ifdef __CUDA_ARCH__
      return detail::short_vector_detail::bounded_index(array_, size() - 1);
#else
      return array_[size()-1];
#endif
    }

    __AGENCY_ANNOTATION
    const_reference back() const
    {
#ifdef __CUDA_ARCH__
      return detail::short_vector_detail::bounded_index(array_, size() - 1);
#else
      return array_[size()-1];
#endif
    }

    __AGENCY_ANNOTATION
    optional<value_type> back_or_none() const
    {
      //return empty() ? nullopt : make_optional(back());
      
      // XXX this requires fewer registers than the equivalent above 
      //     but depends on the knowledge that the implementation of back()
      //     returns a reference to an actual memory location even when size() == 0
      auto result = make_optional(back());
      if(empty()) result = nullopt;
      return result;
    }

    __AGENCY_ANNOTATION
    T* data()
    {
      return array_.data();
    }

    __AGENCY_ANNOTATION
    const T* data() const
    {
      return array_.data();
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
      return size() == size_type(0);
    }

    __AGENCY_ANNOTATION
    constexpr size_type size() const
    {
      return size_;
    }

    __AGENCY_ANNOTATION
    constexpr size_type max_size() const
    {
      return N;
    }

    __AGENCY_ANNOTATION
    void fill(const T& value)
    {
      for(auto& e : *this)
      {
        e = value;
      }
    }

    //__AGENCY_ANNOTATION
    //void swap(short_vector& other)
    //{
    //  for(size_type i = 0; i < size(); ++i)
    //  {
    //    agency::detail::adl_swap((*this)[i], other[i]);
    //  }
    //}

  private:
    array<value_type, N> array_;
    size_type size_;
};


template<class T, std::size_t N>
__AGENCY_ANNOTATION
bool operator==(const short_vector<T,N>& lhs,  const short_vector<T,N>& rhs)
{
  if(lhs.size() != rhs.size()) return false;

  for(std::size_t i = 0; i < lhs.size(); ++i)
  {
    if(lhs[i] != rhs[i]) return false;
  }

  return true;
}

// XXX other relational operators here
// XXX get() here?
// XXX tuple specializations here?


template<class T, std::size_t N>
__AGENCY_ANNOTATION
void swap(short_vector<T,N>& a, short_vector<T,N>& b)
{
  a.swap(b);
}


} // end experimental
} // end agency

