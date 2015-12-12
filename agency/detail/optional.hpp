#pragma once

#include <agency/detail/config.hpp>
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
  T1 tmp = std::move(a);
  a = std::move(b);
  b = std::move(tmp);
}


template<
  class T,
  bool use_empty_base_class_optimization =
    std::is_empty<T>::value
#if __cplusplus >= 201402L
    && !std::is_final<T>::value
#endif
>
struct optional_base
{
  typedef typename std::aligned_storage<
    sizeof(T)
  >::type storage_type;
  
  storage_type storage_;
};

template<class T>
struct optional_base<T,true> : T {};


} // end optional_detail


template<class T>
class optional : public optional_detail::optional_base<T>
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
      emplace(*other);
    }

    __AGENCY_ANNOTATION
    optional(optional&& other)
      : contains_value_(false)
    {
      if(other)
      {
        emplace(std::move(*other));
      }
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
        **this = std::forward<U>(value);
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
        *this = *other;
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
        *this = std::move(*other);
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

      new (operator->()) T(std::forward<Args>(args)...) ;
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
      return operator*();
    }

    __AGENCY_ANNOTATION
    const T& value() const &
    {
      // XXX should check contains_value_ and throw otherwise
      return operator*();
    }

    __AGENCY_ANNOTATION
    T&& value() &&
    {
      // XXX should check contains_value_ and throw otherwise
      return std::move(operator*());
    }

    __AGENCY_ANNOTATION
    void swap(optional& other)
    {
      if(*other)
      {
        if(*this)
        {
          using optional_detail::swap;

          swap(**this, *other);
        }
        else
        {
          emplace(*other);
          other = nullopt;
        }
      }
      else
      {
        if(*this)
        {
          other.emplace(**this);
          *this = nullopt;
        }
        else
        {
          // no effect
        }
      }
    }

    __AGENCY_ANNOTATION
    const T* operator->() const
    {
      return reinterpret_cast<const T*>(this);
    }

    __AGENCY_ANNOTATION
    T* operator->()
    {
      return reinterpret_cast<T*>(this);
    }

    __AGENCY_ANNOTATION
    const T& operator*() const &
    {
      return *operator->();
    }

    __AGENCY_ANNOTATION
    T& operator*() &
    {
      return *operator->();
    }

    __AGENCY_ANNOTATION
    const T&& operator*() const &&
    {
      return *operator->();
    }

    __AGENCY_ANNOTATION
    T&& operator*() &&
    {
      return *operator->();
    }

  private:
    __AGENCY_ANNOTATION
    void clear()
    {
      if(*this)
      {
        operator*().~T();
        contains_value_ = false;
      }
    }

    bool contains_value_;
};


template<class T>
__AGENCY_ANNOTATION
optional<typename std::decay<T>::type> make_optional(T&& value)
{
  return optional<typename std::decay<T>::type>(std::forward<T>(value));
}


} // end detail
} // end agency

