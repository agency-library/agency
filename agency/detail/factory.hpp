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


// call_constructor_factory is a type of Factory
// which creates a T by calling T's constructor with the given Args...
template<class T, class... Args>
class call_constructor_factory
{
  public:
    __AGENCY_ANNOTATION
    call_constructor_factory() : args_() {}

    __AGENCY_ANNOTATION
    call_constructor_factory(const tuple<Args...>& args)
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


template<class T>
__AGENCY_ANNOTATION
call_constructor_factory<T,T> make_call_constructor_factory(const T& arg)
{
  return call_constructor_factory<T,T>{detail::make_tuple(arg)};
}


template<class T, class... Args>
__AGENCY_ANNOTATION
call_constructor_factory<T,typename std::decay<Args>::type...> make_call_constructor_factory(Args&&... args)
{
  return call_constructor_factory<T,typename std::decay<Args>::type...>(agency::detail::make_tuple(std::forward<Args>(args)...));
}


struct unit_factory : call_constructor_factory<unit> {};


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

