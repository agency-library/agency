#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/memory/detail/unique_ptr.hpp>
#include <agency/memory/detail/any_deleter.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


// CUDA's any_small_deleter is just like agency::detail::any_small_deleter
// except that its default state defaults to cuda::detail::default_delete<T>
// rather than agency::detail::default_delete<T>
template<class T>
class any_small_deleter : public agency::detail::any_small_deleter<T>
{
  private:
    using super_t = agency::detail::any_small_deleter<T>;

  public:
    using pointer = typename super_t::pointer;

    // inherit the base class's constructors
    using super_t::super_t;

    // the default constructor sets the concrete deleter to cuda::detail::default_delete<T>
    __AGENCY_ANNOTATION
    any_small_deleter()
      : super_t(cuda::detail::default_delete<T>())
    {}

    // reimplement the copy constructor to ensure that the correct type of object is
    // passed to super_t's copy constructor so that the the correct constructor is chosen
    __AGENCY_ANNOTATION
    any_small_deleter(const any_small_deleter& other)
      : super_t(static_cast<const super_t&>(other))
    {}

    // reimplement the move constructor to ensure that the correct type of object is
    // passed to super_t's move constructor so that the correct constructor is chosen
    __AGENCY_ANNOTATION
    any_small_deleter(any_small_deleter&& other)
      : super_t(static_cast<super_t&&>(other))
    {}

    any_small_deleter& operator=(const any_small_deleter& other)
    {
      super_t::operator=(other);
      return *this;
    }
}; // end any_small_deleter


} // end detail
} // end cuda
} // end agency

