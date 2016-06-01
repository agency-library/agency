#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/factory.hpp>
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
    using value_type = detail::result_of_factory_t<factory_type>;

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


template<size_t scope, class T, class... Args>
__AGENCY_ANNOTATION
detail::shared_parameter<scope,detail::construct<T,Args...>>
  share_at_scope(Args&&... args)
{
  using factory_t = detail::construct<T,Args...>;
  factory_t factory(detail::make_tuple(std::forward<Args>(args)...));
  return detail::shared_parameter<scope,factory_t>(factory);
}


template<size_t scope, class T>
__AGENCY_ANNOTATION
detail::shared_parameter<scope,detail::construct<T,T>>
  share_at_scope(const T& val)
{
  using factory_t = detail::construct<T,T>;
  factory_t factory(detail::make_tuple(val));
  return detail::shared_parameter<scope,factory_t>(factory);
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

