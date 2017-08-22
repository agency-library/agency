#pragma once

#include <agency/detail/config.hpp>
#include <agency/tuple.hpp>
#include <agency/detail/factory.hpp>
#include <agency/detail/type_traits.hpp>
#include <tuple>
#include <utility>
#include <type_traits>

namespace agency
{
namespace detail
{


template<size_t scope, class Factory>
class shared_parameter
{
  public:
    using factory_type = Factory;
    using value_type = result_of_t<factory_type()>;

    __AGENCY_ANNOTATION
    shared_parameter(const factory_type& factory)
      : factory_(factory)
    {}

    __AGENCY_ANNOTATION
    const factory_type& factory() const
    {
      return factory_;
    }

  private:
    factory_type factory_;
};


template<class T> struct is_shared_parameter : std::false_type {};
template<size_t scope, class Factory>
struct is_shared_parameter<shared_parameter<scope,Factory>> : std::true_type {};


template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};


} // end detail


template<size_t scope, class Factory>
__AGENCY_ANNOTATION
detail::shared_parameter<scope,Factory>
  share_at_scope_from_factory(const Factory& factory)
{
  return detail::shared_parameter<scope,Factory>(factory);
}


template<size_t scope, class T, class... Args>
__AGENCY_ANNOTATION
auto share_at_scope(Args&&... args) ->
  decltype(
    agency::share_at_scope_from_factory<scope>(
      detail::make_construct<T>(std::forward<Args>(args)...)
    )
  )
{
  return agency::share_at_scope_from_factory<scope>(
    detail::make_construct<T>(std::forward<Args>(args)...)
  );
}


template<size_t scope, class T>
__AGENCY_ANNOTATION
auto share_at_scope(const T& val) ->
  decltype(
    agency::share_at_scope_from_factory<scope>(
      detail::make_copy_construct(val)
    )
  )
{
  return agency::share_at_scope_from_factory<scope>(
    detail::make_copy_construct(val)
  );
}


template<class T, class... Args>
__AGENCY_ANNOTATION
auto share(Args&&... args) ->
  decltype(agency::share_at_scope<0,T>(std::forward<Args>(args)...))
{
  return agency::share_at_scope<0,T>(std::forward<Args>(args)...);
}


template<class T>
__AGENCY_ANNOTATION
auto share(const T& val) ->
  decltype(agency::share_at_scope<0>(val))
{
  return agency::share_at_scope<0>(val);
}


} // end agency

