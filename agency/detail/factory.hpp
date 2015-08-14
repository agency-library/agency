#pragma once

#include <agency/detail/tuple.hpp>
#include <utility>

namespace agency
{
namespace detail
{


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


struct unit {};


struct unit_factory : factory<unit> {};


} // end detail
} // end agency

