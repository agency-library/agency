#pragma once

// XXX should probably eliminate this header
//     consider renaming it to functional.hpp

#include <agency/detail/config.hpp>
#include <agency/detail/shared_parameter.hpp>
#include <agency/detail/tuple.hpp>

namespace agency
{


template<class T>
struct decay_parameter : std::decay<T> {};


template<class T>
using decay_parameter_t = typename decay_parameter<T>::type;


template<size_t level, class T, class... Args>
struct decay_parameter<
  detail::shared_parameter<level, T, Args...>
>
{
  // shared parameters are passed by reference
  using type = T&;
};


// XXX eliminate me
template<class T>
struct parameter_t : std::decay<T> {};


template<size_t level, class T, class... Args>
struct parameter_t<
  detail::shared_parameter<level, T, Args...>
>
{
  using type = T;
};


template<class... Types>
struct parameter_t<
  detail::tuple<Types...>
>
{
  using type = detail::tuple<typename parameter_t<Types>::type...>;
};


template<class T>
__AGENCY_ANNOTATION
typename parameter_t<T>::type decay_construct(T&& parm)
{
  // this is essentially decay_copy
  return std::forward<T>(parm);
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


// overload decay_construct for shared_parameter
template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(detail::shared_parameter<level,T,Args...>& parm)
{
  return __tu::make_from_tuple<T>(parm.args_);
}


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(const detail::shared_parameter<level,T,Args...>& parm)
{
  return __tu::make_from_tuple<T>(parm.args_);
}


template<size_t level, class T, class... Args>
__AGENCY_ANNOTATION
T decay_construct(detail::shared_parameter<level,T,Args...>&& parm)
{
  return __tu::make_from_tuple<T>(std::move(parm.args_));
}


template<class T>
struct parameter_t;


template<class... Types>
__AGENCY_ANNOTATION
typename parameter_t<detail::tuple<Types...>>::type decay_construct(detail::tuple<Types...>& t);


template<class... Types>
__AGENCY_ANNOTATION
typename parameter_t<detail::tuple<Types...>>::type decay_construct(const detail::tuple<Types...>& t);


template<class... Types>
__AGENCY_ANNOTATION
typename parameter_t<detail::tuple<Types...>>::type decay_construct(detail::tuple<Types...>&& t);


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
typename parameter_t<detail::tuple<Types...>>::type decay_construct(detail::tuple<Types...>& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, t
  );
}


template<class... Types>
__AGENCY_ANNOTATION
typename parameter_t<detail::tuple<Types...>>::type decay_construct(const detail::tuple<Types...>& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, t
  );
}


template<class... Types>
__AGENCY_ANNOTATION
typename parameter_t<detail::tuple<Types...>>::type decay_construct(detail::tuple<Types...>&& t)
{
  return detail::tuple_map(
    detail::call_decay_construct{}, std::move(t)
  );
}


} // end agency

