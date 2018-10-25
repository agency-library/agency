#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/shape.hpp>
#include <agency/coordinate/point.hpp>

#include <initializer_list>
#include <tuple>
#include <type_traits>


// XXX should move lattice into its own header underneath coordinate/
namespace agency
{
namespace detail
{


template<typename T, typename Enable = void>
struct index_size : std::tuple_size<T> {};


template<typename T>
struct index_size<T, typename std::enable_if<std::is_integral<T>::value>::type>
{
  constexpr static size_t value = 1;
};


template<class T> class lattice_iterator;


template<typename Array, typename T>
struct rebind_array;

template<template<typename,size_t> class array_template, size_t N, typename OldT, typename NewT>
struct rebind_array<array_template<OldT,N>, NewT>
{
  using type = array_template<NewT,N>;
};


template<typename Array, typename T>
using rebind_array_t = typename rebind_array<Array,T>::type;


} // end detail


// this class is a lattice, the points of which take on values which are unit-spaced
// T is any orderable (has strict weak <) type with
// operators +, +=, -, -=, *, *=, /, /= such that the rhs's type is regular_grid<T>::index_type
// XXX this type should share a similar interface with ndarray
//     (i.e., these types should model the same Concept)
template<class T>
class lattice
{
  public:
    // XXX should pick a different name for rank
    //     or maybe just eliminate it and rely on .size()
    constexpr static size_t rank = detail::index_size<T>::value;
    using size_type              = size_t;

    using value_type             = T;
    using reference              = value_type;
    using const_reference        = reference;
    using const_iterator         = detail::lattice_iterator<T>;
    using iterator               = const_iterator;

    // returns the value of the smallest lattice point
    __AGENCY_ANNOTATION
    value_type min() const
    {
      return min_;
    }

    // returns the value of the one-past-the-last lattice point
    __AGENCY_ANNOTATION
    value_type max() const
    {
      return max_;
    }

    // returns the number of lattice points along each of this lattice's dimensions
    // chose the name shape instead of extent to jibe with e.g. numpy.ndarray.shape
    __AGENCY_ANNOTATION
    auto shape() const
      -> decltype(
           this->max() - this->min()
         )
    {
      return max() - min();
    }

    // XXX WAR cudafe perf issue
    //using index_type = typename detail::result_of_t<
    //  decltype(&lattice::shape)(lattice)
    //>;
    using index_type = decltype(value_type{} - value_type{});

    // XXX should create a grid empty of points
    lattice() = default;

    // copy from
    lattice(const lattice&) = default;

    // creates a new lattice with min as the first lattice point
    // through max exclusive
    // XXX should probably make this (min, shape) instead
    //     otherwise we'd have to require that min[i] <= max[i]
    __AGENCY_ANNOTATION
    lattice(const value_type& min, const value_type& max)
      : min_(min), max_(max)
    {}

    template<class Size1, class... Sizes,
             typename = typename std::enable_if<
               detail::conjunction<
                 std::is_convertible<Size1,size_t>, std::is_convertible<Sizes,size_t>...
               >::value &&
               sizeof...(Sizes) == (rank - 1)
             >::type 
            >
    __AGENCY_ANNOTATION
    lattice(const Size1& dimension1, const Sizes&... dimensions)
      : lattice(index_type{static_cast<size_t>(dimension1), static_cast<size_t>(dimensions)...})
    {}

    __AGENCY_ANNOTATION
    lattice(const index_type& dimensions)
      : lattice(index_type{}, dimensions)
    {}

    // XXX upon c++14, assert that the intializer_list is of the correct size
    template<class Size,
             class = typename std::enable_if<
               std::is_constructible<index_type, std::initializer_list<Size>>::value
             >::type>
    __AGENCY_ANNOTATION
    lattice(std::initializer_list<Size> dimensions)
      : lattice(index_type{dimensions})
    {}

    // returns whether or not p is the value of a lattice point
    __AGENCY_ANNOTATION
    bool contains(const value_type& p) const
    {
      return contains(p, std::is_arithmetic<value_type>());
    }

    // returns the number of lattice points
    __AGENCY_ANNOTATION
    size_type size() const
    {
      return detail::index_space_size(shape());
    }

    __AGENCY_ANNOTATION
    bool empty() const
    {
      return min() == max();
    }

    // returns the value of the (i,j,k,...)th lattice point
    __AGENCY_ANNOTATION
    const_reference operator[](const index_type& idx) const
    {
      return min() + idx;
    }

    // returns the value of the ith lattice point in lexicographic order
    template<class Size,
             typename std::enable_if<
               (rank > 1) &&
               std::is_convertible<Size,size_type>::value
             >::type
            >
    __AGENCY_ANNOTATION
    const_reference operator[](Size idx) const
    {
      return begin()[idx];
    }

    // reshape does not move the origin
    __AGENCY_ANNOTATION
    void reshape(const index_type& dimensions)
    {
      max_ = min() + dimensions;
    }

    // reshape does not move the origin
    template<class... Size,
             typename = typename std::enable_if<
               detail::conjunction<
                 std::is_convertible<Size,size_t>...
               >::value &&
               sizeof...(Size) == rank
             >::type 
            >
    __AGENCY_ANNOTATION
    void reshape(const Size&... dimensions)
    {
      reshape(index_type{static_cast<size_t>(dimensions)...});
    }

    __AGENCY_ANNOTATION
    const_iterator begin() const
    {
      return detail::lattice_iterator<value_type>(*this);
    }

    __AGENCY_ANNOTATION
    const_iterator end() const
    {
      return detail::lattice_iterator<value_type>(*this, detail::lattice_iterator<value_type>::past_the_end(*this));
    }

  private:
    // T is point-like case
    __AGENCY_ANNOTATION
    bool contains(const value_type& x, std::false_type) const
    {
      bool result = true;

      for(size_t dim = 0; dim != rank; ++dim)
      {
        result = result && min()[dim] <= x[dim];
        result = result && x[dim] < max()[dim];
      }

      return result;
    }

    // T is scalar case
    __AGENCY_ANNOTATION
    bool contains(const value_type& x, std::true_type) const
    {
      return min() <= x && x < max();
    }

    value_type min_, max_;
};


template<class T>
__AGENCY_ANNOTATION
lattice<T> make_lattice(const T& max)
{
  return {max};
}


namespace detail
{


template<class T>
class lattice_iterator
  : public std::iterator<
      std::random_access_iterator_tag,
      T,
      std::ptrdiff_t,
      void, // XXX implement this
      T
    >
{
  private:
    using super_t = std::iterator<
      std::random_access_iterator_tag,
      T,
      std::ptrdiff_t,
      void,
      T
    >;

    static constexpr size_t rank = lattice<T>::rank;

  public:
    using typename super_t::value_type;
    using typename super_t::reference;
    using typename super_t::difference_type;

    __AGENCY_ANNOTATION
    explicit lattice_iterator(const lattice<T>& domain)
      : domain_(domain),
        current_(domain_.min())
    {}

    __AGENCY_ANNOTATION
    explicit lattice_iterator(const lattice<T>& domain, T current)
      : domain_(domain),
        current_(current)
    {}

    __AGENCY_ANNOTATION
    reference operator*() const
    {
      return current_;
    }

    __AGENCY_ANNOTATION
    lattice_iterator& operator++()
    {
      return increment(std::is_arithmetic<T>());
    }

    __AGENCY_ANNOTATION
    lattice_iterator operator++(int)
    {
      lattice_iterator result = *this;
      ++(*this);
      return result;
    }

    __AGENCY_ANNOTATION
    lattice_iterator& operator--()
    {
      return decrement(std::is_arithmetic<T>());
    }

    __AGENCY_ANNOTATION
    lattice_iterator operator--(int)
    {
      lattice_iterator result = *this;
      --(*this);
      return result;
    }

    __AGENCY_ANNOTATION
    lattice_iterator operator+(difference_type n) const
    {
      lattice_iterator result{*this};
      return result += n;
    }

    __AGENCY_ANNOTATION
    lattice_iterator& operator+=(difference_type n)
    {
      return advance(n, std::is_arithmetic<T>());
    }

    __AGENCY_ANNOTATION
    lattice_iterator& operator-=(difference_type n)
    {
      return *this += -n;
    }

    __AGENCY_ANNOTATION
    lattice_iterator operator-(difference_type n) const
    {
      lattice_iterator result{*this};
      return result -= n;
    }

    __AGENCY_ANNOTATION
    difference_type operator-(const lattice_iterator& rhs) const
    {
      return linearize() - rhs.linearize();
    }

    __AGENCY_ANNOTATION
    bool operator==(const lattice_iterator& rhs) const
    {
      return current_ == rhs.current_;
    }

    __AGENCY_ANNOTATION
    bool operator!=(const lattice_iterator& rhs) const
    {
      return !(*this == rhs);
    }

    __AGENCY_ANNOTATION
    bool operator<(const lattice_iterator& rhs) const
    {
      return current_ < rhs.current_;
    }

    __AGENCY_ANNOTATION
    bool operator<=(const lattice_iterator& rhs) const
    {
      return !(rhs < *this);
    }

    __AGENCY_ANNOTATION
    bool operator>(const lattice_iterator& rhs) const
    {
      return rhs < *this;
    }

    __AGENCY_ANNOTATION
    bool operator>=(const lattice_iterator &rhs) const
    {
      return !(rhs > *this);
    }

    __AGENCY_ANNOTATION
    static T past_the_end(const lattice<T>& domain)
    {
      return past_the_end(domain, std::is_arithmetic<T>());
    }

  private:
    // point-like case
    __AGENCY_ANNOTATION
    lattice_iterator& increment(std::false_type)
    {
      T min = domain_.min();
      T max = domain_.max();

      for(int i = rank; i-- > 0;)
      {
        ++current_[i];

        if(min[i] <= current_[i] && current_[i] < max[i])
        {
          return *this;
        }
        else if(i > 0)
        {
          // don't roll the final dimension over to the origin
          current_[i] = min[i];
        }
      }

      return *this;
    }

    // scalar case
    __AGENCY_ANNOTATION
    lattice_iterator& increment(std::true_type)
    {
      ++current_;
      return *this;
    }

    // point-like case
    __AGENCY_ANNOTATION
    lattice_iterator& decrement(std::false_type)
    {
      T min = domain_.min();
      T max = domain_.max();

      for(int i = rank; i-- > 0;)
      {
        --current_[i];

        if(min[i] <= current_[i])
        {
          return *this;
        }
        else
        {
          current_[i] = max[i] - 1;
        }
      }

      return *this;
    }

    // scalar case
    __AGENCY_ANNOTATION
    lattice_iterator& decrement(std::true_type)
    {
      --current_;
      return *this;
    }

    // point-like case
    __AGENCY_ANNOTATION
    lattice_iterator& advance(difference_type n, std::false_type)
    {
      difference_type idx = linearize() + n;

      auto s = stride();

      for(size_t i = 0; i < rank; ++i)
      {
        current_[i] = domain_.min()[i] + idx / s[i];
        idx %= s[i];
      }

      return *this;
    }

    // scalar case
    __AGENCY_ANNOTATION
    lattice_iterator& advance(difference_type n, std::true_type)
    {
      current_ += n;
      return *this;
    }

    __AGENCY_ANNOTATION
    point<difference_type,rank> stride() const
    {
      point<difference_type,rank> result;
      result[rank - 1] = 1;

      for(int i = rank - 1; i-- > 0;)
      {
        // accumulate the stride of the lower dimension
        result[i] = result[i+1] * domain_.shape()[i];
      }

      return result;
    }

    __AGENCY_ANNOTATION
    difference_type linearize() const
    {
      return linearize(std::is_arithmetic<T>());
    }

    // point-like case
    __AGENCY_ANNOTATION
    difference_type linearize(std::false_type) const
    {
      if(is_past_the_end())
      {
        return domain_.size();
      }

      // subtract grid min from current to get
      // 0-based indices along each axis
      T idx = current_ - domain_.min();

      difference_type multiplier = 1;
      difference_type result = 0;

      for(int i = rank; i-- > 0; )
      {
        result += multiplier * idx[i];
        multiplier *= domain_.shape()[i];
      }

      return result;
    }

    // scalar case
    __AGENCY_ANNOTATION
    difference_type linearize(std::true_type) const
    {
      return current_;
    }

    // point-like case
    __AGENCY_ANNOTATION
    static T past_the_end(const lattice<T>& domain, std::false_type)
    {
      T result = domain.min();
      result[0] = domain.max()[0];
      return result;
    }

    // scalar case
    __AGENCY_ANNOTATION
    static T past_the_end(const lattice<T>& domain, std::true_type)
    {
      return domain.max();
    }

    __AGENCY_ANNOTATION
    bool is_past_the_end() const
    {
      return !(current_[0] < domain_.max()[0]);
    }

    lattice<T> domain_;
    T current_;
};


} // end detail
} // end agency

