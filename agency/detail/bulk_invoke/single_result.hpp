#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/optional.hpp>
#include <utility>
#include <tuple>
#include <type_traits>


namespace agency
{


template<class T>
class single_result;


template<class T>
__AGENCY_ANNOTATION
single_result<T> no_result();


template<class T>
class single_result : public detail::optional<T>
{
  private:
    using super_t = detail::optional<T>;

  public:
    using result_type = T;

    __AGENCY_ANNOTATION
    single_result(single_result&& other)
      : super_t(std::move(other))
    {}

    __AGENCY_ANNOTATION
    single_result(const T& result)
      : super_t(result)
    {}

    __AGENCY_ANNOTATION
    single_result(T&& result)
      : super_t(std::move(result))
    {}

    __AGENCY_ANNOTATION
    single_result(const decltype(std::ignore)&)
      : single_result()
    {}

  private:
    __AGENCY_ANNOTATION
    single_result()
      : super_t(detail::nullopt)
    {}

    template<class U>
    __AGENCY_ANNOTATION
    friend single_result<U> no_result();
};


template<class T>
__AGENCY_ANNOTATION
single_result<T> no_result()
{
  return single_result<T>();
} // end no_result()


namespace detail
{


template<class T>
struct is_single_result : std::false_type {};

template<class T>
struct is_single_result<single_result<T>> : std::true_type {};


template<class T, class Shape>
class single_result_container
{
  private:
    // XXX not making this a base class might make it difficult to cast 
    //     future<single_result_container<T>> to future<T>
    T element_;

  public:
    using shape_type = Shape;

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    single_result_container()
      : element_{}
    {}

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    single_result_container(const single_result_container& other)
      : element_(other.element_)
    {}

    // XXX we might prefer to make this a template and do enable_if<is_shape<T>> here,
    //     but g++ < 5 cannot handle it
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    single_result_container(const shape_type&)
      : single_result_container()
    {}

    template<class Index>
    __AGENCY_ANNOTATION
    single_result_container& operator[](const Index&)
    {
      return *this;
    }

    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    void operator=(single_result<T>&& result)
    {
      if(result)
      {
        element_ = std::move(*result);
      }
    }

    __AGENCY_ANNOTATION
    operator T& () &
    {
      return element_;
    }

    __AGENCY_ANNOTATION
    operator const T& () const &
    {
      return element_;
    }

    // XXX this may be the only conversion we want
    __AGENCY_ANNOTATION
    operator const T&& () &&
    {
      return std::move(element_);
    }
};


} // end detail
} // end agency

