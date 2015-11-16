#pragma once

#include <agency/detail/tuple.hpp>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<class Factory>
struct result_of_factory : std::result_of<Factory()> {};


template<class Factory>
using result_of_factory_t = typename result_of_factory<Factory>::type;


template<class T, class... Args>
class factory
{
  public:
    __AGENCY_ANNOTATION
    factory() : args_() {}

    __AGENCY_ANNOTATION
    factory(const tuple<Args...>& args)
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
factory<T,T> make_factory(const T& arg)
{
  return factory<T,T>{detail::make_tuple(arg)};
}


template<class T, class... Args>
__AGENCY_ANNOTATION
factory<T,typename std::decay<Args>::type...> make_factory(Args&&... args)
{
  return factory<T,typename std::decay<Args>::type...>(agency::detail::make_tuple(std::forward<Args>(args)...));
}


struct unit {};


struct unit_factory : factory<unit> {};


template<class... Factories>
struct zip_factory
{
  tuple<Factories...> factory_tuple_;

  __AGENCY_ANNOTATION
  zip_factory(const tuple<Factories...>& factories) : factory_tuple_(factories) {}


  template<size_t... Indices>
  __AGENCY_ANNOTATION
  agency::detail::tuple<
    typename std::result_of<Factories()>::type...
  >
    impl(agency::detail::index_sequence<Indices...>)
  {
    return agency::detail::make_tuple(detail::get<Indices>(factory_tuple_)()...);
  }

  __AGENCY_ANNOTATION
  agency::detail::tuple<
    typename std::result_of<Factories()>::type...
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

