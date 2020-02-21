#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/requires.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/shape.hpp>
#include <agency/coordinate/point.hpp>

#include <initializer_list>
#include <tuple>
#include <type_traits>


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


template<class Index> class lattice_iterator;


} // end detail


// this class is a lattice, the points of which take on values which are unit-spaced
// Index is any orderable (has strict weak <) type with operators +, +=, -, -=, *, *=, /, /=
template<class Index>
class lattice
{
  public:
    using size_type  = std::size_t;
    using value_type = Index;
    using reference  = value_type;
    using iterator   = detail::lattice_iterator<Index>;

    using index_type = value_type;
    using shape_type = index_type;

    // returns the number of dimensions spanned by this lattice
    __AGENCY_ANNOTATION
    static constexpr size_type rank()
    {
      return detail::index_size<Index>::value;
    }

    // default constructor
    lattice() = default;

    // copy constructor
    lattice(const lattice&) = default;

    // (origin, shape) constructor
    // creates a new lattice with at the origin at zero with the given shape
    __AGENCY_ANNOTATION
    lattice(const index_type& origin, const shape_type& shape)
      : origin_(origin), shape_(shape)
    {}

    // shape constructor
    // creates a new lattice with at the origin at zero with the given shape
    __AGENCY_ANNOTATION
    explicit lattice(const shape_type& shape)
      : lattice(index_type{}, shape)
    {}

    // variadic constructor
    template<class Size1, class... Sizes,
             __AGENCY_REQUIRES(
               detail::conjunction<std::is_convertible<Size1,std::size_t>, std::is_convertible<Sizes,std::size_t>...>::value
             ),
             __AGENCY_REQUIRES(sizeof...(Sizes) == (rank() - 1))
            >
    __AGENCY_ANNOTATION
    explicit lattice(const Size1& dimension1, const Sizes&... dimensions)
      : lattice(index_type{static_cast<size_t>(dimension1), static_cast<size_t>(dimensions)...})
    {}

    // XXX upon c++14, assert that the intializer_list is of the correct size
    template<class Size,
             __AGENCY_REQUIRES(std::is_constructible<index_type, std::initializer_list<Size>>::value)
            >
    __AGENCY_ANNOTATION
    lattice(std::initializer_list<Size> dimensions)
      : lattice(index_type{dimensions})
    {}


    // returns the value of the smallest lattice point
    __AGENCY_ANNOTATION
    index_type origin() const
    {
      return origin_;
    }

    // returns the number of lattice points along each of this lattice's dimensions
    // chose the name shape instead of extent to jibe with e.g. numpy.ndarray.shape
    __AGENCY_ANNOTATION
    shape_type shape() const
    {
      return shape_;
    }

    // returns whether or not p is the value of a lattice point
    __AGENCY_ANNOTATION
    bool contains(const index_type& p) const
    {
      return origin() <= p and p < (origin() + shape());
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
      return shape() == shape_type{};
    }

    // returns the value of the (i,j,k,...)th lattice point
    __AGENCY_ANNOTATION
    value_type operator[](const index_type& idx) const
    {
      return origin() + idx;
    }

    // returns the value of the ith lattice point in lexicographic order
    template<class Size,
             __AGENCY_REQUIRES(rank() > 1),
             __AGENCY_REQUIRES(std::is_convertible<Size,size_type>::value)
            >
    __AGENCY_ANNOTATION
    value_type operator[](Size idx) const
    {
      return begin()[idx];
    }

    // reshape does not move the origin
    __AGENCY_ANNOTATION
    void reshape(const shape_type& shape)
    {
      shape_ = shape;
    }

    // reshape does not move the origin
    template<class... Size,
             __AGENCY_REQUIRES(detail::conjunction<std::is_convertible<Size,size_t>...>::value),
             __AGENCY_REQUIRES(sizeof...(Size) == rank())
            >
    __AGENCY_ANNOTATION
    void reshape(const Size&... dimensions)
    {
      reshape(shape_type{static_cast<size_t>(dimensions)...});
    }

    __AGENCY_ANNOTATION
    iterator begin() const
    {
      return detail::lattice_iterator<index_type>(*this);
    }

    __AGENCY_ANNOTATION
    iterator end() const
    {
      return detail::lattice_iterator<index_type>(*this, detail::lattice_iterator<index_type>::past_the_end(*this));
    }

  private:
    index_type origin_;
    shape_type shape_;
};


template<class Shape>
__AGENCY_ANNOTATION
lattice<Shape> make_lattice(const Shape& shape)
{
  return {shape};
}


namespace detail
{


template<class Index>
class lattice_iterator
{
  public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = Index;
    using difference_type = std::ptrdiff_t;
    using pointer = void;
    using reference = value_type;

    __AGENCY_ANNOTATION
    explicit lattice_iterator(const lattice<Index>& domain)
      : domain_(domain),
        current_(domain_.origin())
    {}

    __AGENCY_ANNOTATION
    explicit lattice_iterator(const lattice<Index>& domain, Index current)
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
      return increment();
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
      return decrement();
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
      return advance(n, std::is_arithmetic<Index>());
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
    static Index past_the_end(const lattice<Index>& domain)
    {
      return past_the_end(domain, std::is_arithmetic<Index>());
    }

  private:
    // point-like case
    template<__AGENCY_REQUIRES(!std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& increment()
    {
      Index begin = domain_.origin();
      Index end   = begin + domain_.shape();

      for(int i = domain_.rank(); i-- > 0;)
      {
        ++current_[i];

        if(begin[i] <= current_[i] and current_[i] < end[i])
        {
          return *this;
        }
        else if(i > 0)
        {
          // don't roll the final dimension over to the origin
          current_[i] = begin[i];
        }
      }

      return *this;
    }

    // scalar case
    template<__AGENCY_REQUIRES(std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& increment()
    {
      ++current_;
      return *this;
    }

    // point-like case
    template<__AGENCY_REQUIRES(!std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& decrement()
    {
      Index begin = domain_.origin();
      Index end   = begin + domain_.shape();

      for(int i = domain_.rank(); i-- > 0;)
      {
        --current_[i];

        if(begin[i] <= current_[i])
        {
          return *this;
        }
        else
        {
          current_[i] = end[i] - 1;
        }
      }

      return *this;
    }

    // scalar case
    template<__AGENCY_REQUIRES(std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& decrement()
    {
      --current_;
      return *this;
    }

    // point-like case
    template<__AGENCY_REQUIRES(!std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& advance(difference_type n)
    {
      difference_type idx = linearize() + n;

      auto s = stride();

      for(size_t i = 0; i < domain_.rank(); ++i)
      {
        current_[i] = domain_.origin()[i] + idx / s[i];
        idx %= s[i];
      }

      return *this;
    }

    // scalar case
    template<__AGENCY_REQUIRES(std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    lattice_iterator& advance(difference_type n)
    {
      current_ += n;
      return *this;
    }

    __AGENCY_ANNOTATION
    point<difference_type,lattice<Index>::rank()> stride() const
    {
      constexpr std::size_t rank = lattice<Index>::rank();

      point<difference_type,rank> result;
      result[rank - 1] = 1;

      for(int i = rank - 1; i-- > 0;)
      {
        // accumulate the stride of the lower dimension
        result[i] = result[i+1] * domain_.shape()[i];
      }

      return result;
    }

    // point-like case
    template<__AGENCY_REQUIRES(!std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    difference_type linearize() const
    {
      if(is_past_the_end())
      {
        return domain_.size();
      }

      // subtract the origin from current to get
      // 0-based indices along each axis
      Index idx = current_ - domain_.origin();

      difference_type multiplier = 1;
      difference_type result = 0;

      for(int i = domain_.rank(); i-- > 0; )
      {
        result += multiplier * idx[i];
        multiplier *= domain_.shape()[i];
      }

      return result;
    }

    // scalar case
    template<__AGENCY_REQUIRES(std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    difference_type linearize() const
    {
      return current_;
    }

    // point-like case
    template<__AGENCY_REQUIRES(!std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    static Index past_the_end(const lattice<Index>& domain)
    {
      Index result = domain.origin();
      result[0] = domain.origin() + domain.shape()[0];
      return result;
    }

    // scalar case
    template<__AGENCY_REQUIRES(std::is_arithmetic<Index>::value)>
    __AGENCY_ANNOTATION
    static Index past_the_end(const lattice<Index>& domain)
    {
      return domain.origin() + domain.shape();
    }

    __AGENCY_ANNOTATION
    bool is_past_the_end() const
    {
      auto end = domain_.origin() + domain_.shape();
      return !(current_[0] < end[0]);
    }

    lattice<Index> domain_;
    Index current_;
};


} // end detail
} // end agency

