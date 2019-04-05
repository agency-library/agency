#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


// construct is a type of Factory
// which creates a T by calling T's constructor with the given Args...
template<class T, class... Args>
class construct
{
  public:
    __AGENCY_ANNOTATION
    construct() : args_() {}

    __AGENCY_ANNOTATION
    construct(const tuple<Args...>& args)
      : args_(args)
    {}

    __agency_exec_check_disable__
    template<size_t... Indices>
    __AGENCY_ANNOTATION
    T impl(index_sequence<Indices...>) const &
    {
      return T(agency::get<Indices>(args_)...);
    }

    __agency_exec_check_disable__
    template<size_t... Indices>
    __AGENCY_ANNOTATION
    T impl(index_sequence<Indices...>) &&
    {
      return T(agency::get<Indices>(std::move(args_))...);
    }

    __AGENCY_ANNOTATION
    T operator()() const &
    {
      return impl(make_index_sequence<sizeof...(Args)>());
    }

    __AGENCY_ANNOTATION
    T operator()() &&
    {
      return std::move(*this).impl(make_index_sequence<sizeof...(Args)>());
    }

  private:
    tuple<Args...> args_;
};


template<class T, class... Args>
__AGENCY_ANNOTATION
construct<T,typename std::decay<Args>::type...> make_construct(Args&&... args)
{
  return construct<T,typename std::decay<Args>::type...>(agency::make_tuple(std::forward<Args>(args)...));
}


template<class T>
__AGENCY_ANNOTATION
construct<T,T> make_copy_construct(const T& arg)
{
  return make_construct<T>(arg);
}


struct unit_factory : construct<unit> {};


// a moving_factory is a factory which moves an object when it is called
template<class T>
class moving_factory
{
  public:
    moving_factory(moving_factory&& other) = default;

    // this constructor moves other's value into value_
    // so, it acts like a move constructor
    __AGENCY_ANNOTATION
    moving_factory(const moving_factory& other)
      : value_(std::move(other.value_))
    {}

    __agency_exec_check_disable__
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<T,U&&>::value
             >::type>
    __AGENCY_ANNOTATION
    moving_factory(U&& value)
      : value_(std::forward<U>(value))
    {}

    __AGENCY_ANNOTATION
    T operator()() const
    {
      return std::move(value_);
    }

  private:
    mutable T value_;
};


template<class T>
__AGENCY_ANNOTATION
moving_factory<decay_t<T>> make_moving_factory(T&& value)
{
  return moving_factory<decay_t<T>>(std::forward<T>(value));
}


// a zip_factory is a type of Factory which takes a list of Factories
// and creates a tuple whose elements are the results of the given Factories
template<class... Factories>
struct zip_factory
{
  tuple<Factories...> factory_tuple_;

  __AGENCY_ANNOTATION
  zip_factory(const tuple<Factories...>& factories) : factory_tuple_(factories) {}


  template<size_t... Indices>
  __AGENCY_ANNOTATION
  agency::tuple<
    result_of_t<Factories()>...
  >
    impl(agency::detail::index_sequence<Indices...>)
  {
    return agency::make_tuple(agency::get<Indices>(factory_tuple_)()...);
  }

  __AGENCY_ANNOTATION
  agency::tuple<
    result_of_t<Factories()>...
  >
    operator()()
  {
    return impl(index_sequence_for<Factories...>());
  }
};


template<class... Factories>
__AGENCY_ANNOTATION
zip_factory<Factories...> make_zip_factory(const tuple<Factories...>& factory_tuple)
{
  return zip_factory<Factories...>(factory_tuple);
}


} // end detail
} // end agency

