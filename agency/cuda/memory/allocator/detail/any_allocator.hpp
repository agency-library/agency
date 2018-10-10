#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/allocator/detail/any_allocator.hpp>

namespace agency
{
namespace cuda
{
namespace detail
{


// CUDA's any_small_allocator is just like agency::detail::any_small_allocator
// except that its default-constructed state is to contain cuda::allocator<T>
// rather than std::allocator<T>
template<class T>
class any_small_allocator : public agency::detail::any_small_allocator<T>
{
  private:
    using super_t = agency::detail::any_small_allocator<T>;

  public:
    // inherit the base class's constructors
    using super_t::super_t;

    // the default constructor sets the concrete allocator to cuda::allocator<T>
    __AGENCY_ANNOTATION
    any_small_allocator()
      : super_t(cuda::allocator<T>())
    {}

    // reimplement the copy constructor to ensure that the correct type of object is
    // passed to super_t's copy constructor so that the the correct constructor is chosen
    __AGENCY_ANNOTATION
    any_small_allocator(const any_small_allocator& other)
      : super_t(static_cast<const super_t&>(other))
    {}

    // reimplement the move constructor to ensure that the correct type of object is
    // passed to super_t's move constructor so that the correct constructor is chosen
    __AGENCY_ANNOTATION
    any_small_allocator(any_small_allocator&& other)
      : super_t(static_cast<super_t&&>(other))
    {}

    any_small_allocator& operator=(const any_small_allocator& other)
    {
      super_t::operator=(other);
      return *this;
    }
}; // end any_small_allocator


} // end detail
} // end cuda
} // end agency

