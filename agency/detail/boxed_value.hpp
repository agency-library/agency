#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/unique_ptr.hpp>
#include <memory>

namespace agency
{
namespace detail
{


template<class T, class Alloc = std::allocator<T>>
class boxed_value
{
  public:
    using allocator_type = typename std::allocator_traits<Alloc>::template rebind_alloc<T>;
    using value_type = typename allocator_type::value_type;

    __AGENCY_ANNOTATION
    boxed_value()
      : boxed_value(value_type{})
    {}

    __AGENCY_ANNOTATION
    boxed_value(const boxed_value& other)
      : boxed_value(other.value())
    {}

    __AGENCY_ANNOTATION
    boxed_value(boxed_value&& other)
      : boxed_value(std::move(other.value()))
    {}

    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args&&...>::value
             >::type>
    __AGENCY_ANNOTATION
    explicit boxed_value(Args... args)
      : data_(agency::detail::allocate_unique<T>(allocator_type(), std::forward<Args>(args)...))
    {}

    __AGENCY_ANNOTATION
    value_type& value() &
    {
      return *data_;
    }

    __AGENCY_ANNOTATION
    const value_type& value() const &
    {
      return *data_;
    }

    __AGENCY_ANNOTATION
    value_type&& value() &&
    {
      return std::move(*data_);
    }

    __AGENCY_ANNOTATION
    const value_type&& value() const &&
    {
      return std::move(*data_);
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<value_type,U&&>::value
             >::type>
    __AGENCY_ANNOTATION
    boxed_value& operator=(U&& other)
    {
      value() = std::forward<U>(other);
      return *this;
    }

  private:
    agency::detail::unique_ptr<T,agency::detail::deleter<allocator_type>> data_;
};


// when the allocator is std::allocator<T>, we can just put this on the stack
template<class T, class OtherT>
class boxed_value<T,std::allocator<OtherT>>
{
  public:
    using allocator_type = std::allocator<T>;
    using value_type = typename allocator_type::value_type;

    __AGENCY_ANNOTATION
    boxed_value()
      : boxed_value(value_type{})
    {}

    __AGENCY_ANNOTATION
    boxed_value(const boxed_value& other)
      : boxed_value(other.value())
    {}

    __AGENCY_ANNOTATION
    boxed_value(boxed_value&& other)
      : boxed_value(std::move(other.value_))
    {}

    template<class... Args,
             class = typename std::enable_if<
               std::is_constructible<T,Args&&...>::value
             >::type>
    __AGENCY_ANNOTATION
    explicit boxed_value(Args&&... args)
      : value_(std::forward<Args>(args)...)
    {}

    __AGENCY_ANNOTATION
    value_type& value()
    {
      return value_;
    }

    __AGENCY_ANNOTATION
    const value_type& value() const
    {
      return value_;
    }

    template<class U,
             class = typename std::enable_if<
               std::is_assignable<value_type,U&&>::value
             >::type>
    __AGENCY_ANNOTATION
    boxed_value& operator=(U&& other)
    {
      value() = std::forward<U>(other);
      return *this;
    }

  private:
    value_type value_;
};


template<class T, class Alloc, class... Args>
__AGENCY_ANNOTATION
boxed_value<T,Alloc> allocate_boxed(const Alloc&, Args&&... args)
{
  return boxed_value<T,Alloc>(std::forward<Args>(args)...);
}


} // end detail
} // end agency

