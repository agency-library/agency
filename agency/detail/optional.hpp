#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/uninitialized.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


struct nullopt_t {};
constexpr nullopt_t nullopt{};


namespace optional_detail
{


__agency_hd_warning_disable__
template<class T1, class T2>
__AGENCY_ANNOTATION
static void swap(T1& a, T2& b)
{
  T1 temp = a;
  a = b;
  b = temp;
}


} // end optional_detail


template<class T>
class optional
{
  public:
    __AGENCY_ANNOTATION
    optional(nullopt_t) : contains_value_{false} {}

    __AGENCY_ANNOTATION
    optional() : optional(nullopt) {}

    __AGENCY_ANNOTATION
    optional(const optional& other)
      : contains_value_(other.contains_value_)
    {
      emplace(other.value_);
    }

    __AGENCY_ANNOTATION
    optional(optional&& other)
      : contains_value_(false)
    {
      emplace(std::move(other.value_));
    }

    __AGENCY_ANNOTATION
    optional(const T& value)
      : contains_value_(false)
    {
      emplace(value);
    }

    __AGENCY_ANNOTATION
    optional(T&& value)
      : contains_value_(false)
    {
      emplace(std::move(value));
    }

    __AGENCY_ANNOTATION
    ~optional()
    {
      clear();
    }

    __AGENCY_ANNOTATION
    optional& operator=(nullopt_t)
    {
      clear();
      return *this;
    }

    template<class U,
             class = typename std::enable_if<
               std::is_same<typename std::decay<U>::type,T>::value
             >::type>
    __AGENCY_ANNOTATION
    optional& operator=(U&& value)
    {
      if(*this)
      {
        value_.get() = std::forward<U>(value);
      }
      else
      {
        emplace(std::forward<U>(value));
      }

      return *this;
    }

    __AGENCY_ANNOTATION
    optional& operator=(const optional& other)
    {
      if(other)
      {
        *this = other.value_;
      }
      else
      {
        *this = nullopt;
      }

      return *this;
    }

    __AGENCY_ANNOTATION
    optional& operator=(optional&& other)
    {
      if(other)
      {
        *this = std::move(other.value_);
      }
      else
      {
        *this = nullopt;
      }

      return *this;
    }

    template<class... Args>
    __AGENCY_ANNOTATION
    void emplace(Args&&... args)
    {
      clear();

      value_.construct(std::forward<Args>(args)...);
      contains_value_ = true;
    }

    __AGENCY_ANNOTATION
    explicit operator bool() const
    {
      return contains_value_;
    }

    __AGENCY_ANNOTATION
    T& value() &
    {
      // XXX should check contains_value_ and throw otherwise
      return value_.get();
    }

    __AGENCY_ANNOTATION
    const T& value() const &
    {
      // XXX should check contains_value_ and throw otherwise
      return value_.get();
    }

    __AGENCY_ANNOTATION
    T&& value() &&
    {
      // XXX should check contains_value_ and throw otherwise
      return std::move(value_.get());
    }

    __AGENCY_ANNOTATION
    void swap(optional& other)
    {
      if(*other)
      {
        if(*this)
        {
          using optional_detail::swap;

          swap(value_, other.value_);
        }
        else
        {
          emplace(other.value_);
          other = nullopt;
        }
      }
      else
      {
        if(*this)
        {
          other.emplace(value_);
          *this = nullopt;
        }
        else
        {
          // no effect
        }
      }
    }

  private:
    __AGENCY_ANNOTATION
    void clear()
    {
      if(*this)
      {
        value_.destroy();
        contains_value_ = false;
      }
    }

    bool contains_value_;
    uninitialized<T> value_;
};


template<class T>
__AGENCY_ANNOTATION
optional<typename std::decay<T>::type> make_optional(T&& value)
{
  return optional<typename std::decay<T>::type>(std::forward<T>(value));
}


} // end detail
} // end agency
