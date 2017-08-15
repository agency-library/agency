#pragma once

#include <agency/detail/config.hpp>
#include <agency/memory/detail/unique_ptr.hpp>
#include <agency/memory/detail/allocation_deleter.hpp>
#include <agency/tuple.hpp>
#include <memory>
#include <type_traits>


namespace agency
{
namespace detail
{


namespace boxed_value_detail
{


template<class T, class Alloc>
struct use_small_object_optimization : disjunction<
  std::is_same<Alloc,std::allocator<T>>,
  std::is_empty<T>,
  is_empty_tuple<T> 
>
{};


} // end boxed_value_detail


template<class T, class Alloc = std::allocator<T>,
         bool use_optimization = boxed_value_detail::use_small_object_optimization<T,Alloc>::value>
class boxed_value : private std::allocator_traits<Alloc>::template rebind_alloc<T>
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
      : data_(agency::detail::allocate_unique<T>(get_allocator(), std::forward<Args>(args)...))
    {}

    __AGENCY_ANNOTATION
    boxed_value& operator=(const boxed_value& other)
    {
      value() = other.value();
      return *this;
    }

    __AGENCY_ANNOTATION
    boxed_value& operator=(boxed_value&& other)
    {
      value() = std::move(other.value());
      return *this;
    }

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
    __AGENCY_ANNOTATION
    allocator_type& get_allocator()
    {
      return *this;
    }

    agency::detail::unique_ptr<T,agency::detail::allocation_deleter<allocator_type>> data_;
};


// when using the optimization, we put the object on the stack
template<class T, class Alloc>
class boxed_value<T,Alloc,true>
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

