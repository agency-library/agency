#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/unit.hpp>
#include <agency/detail/type_traits.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Factory>
using result_of_factory_t = result_of_t<Factory()>;


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

    // XXX eliminate me!
    __AGENCY_ANNOTATION
    T make() const &
    {
      return __tu::make_from_tuple<T>(args_);
    }

    // XXX eliminate me!
    __AGENCY_ANNOTATION
    T make() &&
    {
      return __tu::make_from_tuple<T>(std::move(args_));
    }

    __AGENCY_ANNOTATION
    T operator()() const &
    {
      return make();
    }

    __AGENCY_ANNOTATION
    T operator()() &&
    {
      return std::move(*this).make();
    }

  private:
    tuple<Args...> args_;
};


template<class T, class... Args>
__AGENCY_ANNOTATION
construct<T,typename std::decay<Args>::type...> make_construct(Args&&... args)
{
  return construct<T,typename std::decay<Args>::type...>(agency::detail::make_tuple(std::forward<Args>(args)...));
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
    // XXX this code causes nvcc 8.0 to produce an error message
    //     
    //__agency_exec_check_disable__
    //template<class U,
    //         class = typename std::enable_if<
    //           std::is_constructible<T,U&&>::value
    //         >::type>
    //__AGENCY_ANNOTATION
    //moving_factory(U&& value)
    //  : value_(std::forward<U>(value))
    //{}
    
    // XXX in order to WAR the nvcc 8.0 error above,
    //     instead of perfectly forwarding the value in,
    //     move construct it into value_ instead.
    __agency_exec_check_disable__
    __AGENCY_ANNOTATION
    moving_factory(T&& value)
      : value_(std::move(value))
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
  agency::detail::tuple<
    result_of_factory_t<Factories>...
  >
    impl(agency::detail::index_sequence<Indices...>)
  {
    return agency::detail::make_tuple(detail::get<Indices>(factory_tuple_)()...);
  }

  __AGENCY_ANNOTATION
  agency::detail::tuple<
    result_of_factory_t<Factories>...
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

