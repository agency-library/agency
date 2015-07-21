#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/tuple.hpp>
#include <utility>
#include <type_traits>

namespace agency
{


template<class T>
struct decay_construct_result : std::decay<T> {};


template<class T>
using decay_construct_result_t = typename decay_construct_result<T>::type;


template<class... Types>
struct decay_construct_result<
  detail::tuple<Types...>
>
{
  using type = detail::tuple<typename decay_construct_result<Types>::type...>;
};


template<class T>
__AGENCY_ANNOTATION
decay_construct_result_t<T> decay_construct(T&& parm)
{
  // this is essentially decay_copy
  return std::forward<T>(parm);
}


namespace detail
{


template<class T, class... Args>
class factory
{
  public:
    __AGENCY_ANNOTATION
    factory(const tuple<Args...>& args)
      : args_(args)
    {}

    __AGENCY_ANNOTATION
    T make() const &
    {
      return __tu::make_from_tuple<T>(args_);
    }

    __AGENCY_ANNOTATION
    T make() &&
    {
      return __tu::make_from_tuple<T>(std::move(args_));
    }

  private:
    tuple<Args...> args_;
};


template<size_t level, class T, class... Args>
struct shared_parameter : public factory<T,Args...>
{
  using factory<T,Args...>::factory;
};


template<class T> struct is_shared_parameter : std::false_type {};
template<size_t level, class T, class... Args>
struct is_shared_parameter<shared_parameter<level,T,Args...>> : std::true_type {};


template<class T>
struct is_shared_parameter_ref
  : std::integral_constant<
      bool,
      (std::is_reference<T>::value && is_shared_parameter<typename std::remove_reference<T>::type>::value)
    >
{};


} // end detail


template<size_t level, class T, class... Args>
struct decay_construct_result<
  detail::shared_parameter<level, T, Args...>
>
{
  using type = T;
};


// overload decay_construct for shared_parameter
template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(detail::shared_parameter<level,T,Args...>& parm)
{
  return parm.make();
}


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(const detail::shared_parameter<level,T,Args...>& parm)
{
  return parm.make();
}


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(detail::shared_parameter<level,T,Args...>&& parm)
{
  return std::move(parm).make();
}


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
detail::shared_parameter<level, T,Args...> share(Args&&... args)
{
  return detail::shared_parameter<level, T,Args...>{detail::make_tuple(std::forward<Args>(args)...)};
}


template<size_t level, class T>
__AGENCY_ANNOTATION
detail::shared_parameter<level,T,T> share(const T& val)
{
  return detail::shared_parameter<level,T,T>{detail::make_tuple(val)};
}


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(detail::tuple<Types...>& t);


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(const detail::tuple<Types...>& t);


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(detail::tuple<Types...>&& t);


namespace detail
{


struct call_decay_construct
{
  template<class T>
  __AGENCY_ANNOTATION
  auto operator()(T&& arg) const
    -> decltype(
         agency::decay_construct(std::forward<T>(arg))
       )
  {
    return agency::decay_construct(std::forward<T>(arg));
  }
};


} // end detail


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(detail::tuple<Types...>& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, t
  );
}


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(const detail::tuple<Types...>& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, t
  );
}


template<class... Types>
__AGENCY_ANNOTATION
decay_construct_result_t<detail::tuple<Types...>> decay_construct(detail::tuple<Types...>&& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, std::move(t)
  );
}


__agency_hd_warning_disable__
template<class F, class... Args>
inline __AGENCY_ANNOTATION
auto invoke(F&& f, Args&&... args) -> 
  decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
  return std::forward<F>(f)(std::forward<Args>(args)...);
};


} // end agency

