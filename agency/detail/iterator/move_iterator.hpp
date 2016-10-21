#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/iterator/forwarding_iterator.hpp>
#include <type_traits>
#include <memory>

namespace agency
{
namespace detail
{


template<class Iterator>
struct move_iterator_reference
{
  using base_reference = typename std::iterator_traits<Iterator>::reference;

  using type = typename std::conditional<
    std::is_reference<base_reference>::value,
    typename std::add_rvalue_reference<
      typename std::decay<base_reference>::type
    >::type,
    base_reference
  >::type;
};

template<class Iterator>
using move_iterator_reference_t = typename move_iterator_reference<Iterator>::type;


template<class Iterator>
class move_iterator : public forwarding_iterator<Iterator, move_iterator_reference_t<Iterator>>
{
  private:
    using super_t = forwarding_iterator<Iterator, move_iterator_reference_t<Iterator>>;

  public:
    move_iterator() = default;

    __AGENCY_ANNOTATION
    explicit move_iterator(Iterator x)
      : super_t(x)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    move_iterator(const move_iterator<U>& other)
      : super_t(other)
    {}

    using difference_type = typename super_t::difference_type;

    __AGENCY_ANNOTATION
    move_iterator operator+(difference_type n) const
    {
      move_iterator result = *this;
      result += n;
      return result;
    }
};


template<class Iterator>
__AGENCY_ANNOTATION
move_iterator<Iterator> make_move_iterator(Iterator i)
{
  return move_iterator<Iterator>(i);
}


} // end detail
} // end agency

