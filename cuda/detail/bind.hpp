#pragma once

#include <type_traits>
#include <utility>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>
#include <thrust/functional.h>
#include "tuple.hpp"


namespace cuda
{
namespace detail
{
namespace bind_detail
{


template<class T>
struct is_placeholder : std::false_type {};


template<unsigned int i>
struct is_placeholder<
  thrust::detail::functional::actor<
    thrust::detail::functional::argument<i>
  >
> : std::true_type {};


template<class T>
using decay_t = agency::detail::decay_t<T>;


__thrust_hd_warning_disable__
template<typename F, typename Tuple, size_t... I>
__host__ __device__
auto apply_impl(F&& f, Tuple&& t, agency::detail::index_sequence<I...>)
  -> decltype(
       std::forward<F>(f)(
         thrust::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return std::forward<F>(f)(
    thrust::get<I>(std::forward<Tuple>(t))...
  );
}


template<typename F, typename Tuple>
__host__ __device__
auto apply(F&& f, Tuple&& t)
  -> decltype(
       apply_impl(
         std::forward<F>(f),
         std::forward<Tuple>(t),
         agency::detail::make_index_sequence<std::tuple_size<decay_t<Tuple>>::value>()
       )
     )
{
  using Indices = agency::detail::make_index_sequence<std::tuple_size<decay_t<Tuple>>::value>;
  return apply_impl(
    std::forward<F>(f),
    std::forward<Tuple>(t),
    Indices()
  );
}


template<class ArgTuple, class BoundArg>
__host__ __device__
auto substitute_arg(ArgTuple&&, BoundArg&& bound_arg,
                    typename std::enable_if<
                      !is_placeholder<decay_t<BoundArg>>::value
                    >::type* = 0)
  -> decltype(
       std::forward<BoundArg>(bound_arg)
     )
{
  return std::forward<BoundArg>(bound_arg);
}


template<unsigned int i>
struct placeholder
  : thrust::detail::functional::actor<
      thrust::detail::functional::argument<i>
    >
{};


template<class T>
struct argument_index
  : std::integral_constant<
      unsigned int, 0
    >
{};


template<unsigned int i>
struct argument_index<
  thrust::detail::functional::actor<
    thrust::detail::functional::argument<i>
  >
>
  : std::integral_constant<
      unsigned int, i
    >
{};


template<class ArgTuple, class BoundArg>
__host__ __device__
auto substitute_arg(ArgTuple&& arg_tuple, const BoundArg&,
                   typename std::enable_if<
                     is_placeholder<decay_t<BoundArg>>::value
                   >::type* = 0)
  -> decltype(
       thrust::get<
         argument_index<BoundArg>::value
       >(std::forward<ArgTuple>(arg_tuple))
     )
{
  const unsigned int idx = argument_index<BoundArg>::value;
  return thrust::get<idx>(std::forward<ArgTuple>(arg_tuple));
}


template<class ArgTuple, class BoundArgTuple, size_t... I>
__host__ __device__
auto substitute_impl(ArgTuple&& arg_tuple, BoundArgTuple&& bound_arg_tuple, agency::detail::index_sequence<I...>)
  -> decltype(
       detail::forward_as_tuple(
         substitute_arg(
           std::forward<ArgTuple>(arg_tuple),
           thrust::get<I>(std::forward<BoundArgTuple>(bound_arg_tuple))
         )...
       )
     )
{
  return detail::forward_as_tuple(
    substitute_arg(
      std::forward<ArgTuple>(arg_tuple),
      thrust::get<I>(std::forward<BoundArgTuple>(bound_arg_tuple))
    )...
  );
}


template<class ArgTuple, class BoundArgTuple>
__host__ __device__
auto substitute(ArgTuple&& arg_tuple, BoundArgTuple&& bound_arg_tuple)
  -> decltype(
       substitute_impl(
         std::forward<ArgTuple>(arg_tuple),
         std::forward<BoundArgTuple>(bound_arg_tuple),
         agency::detail::make_index_sequence<std::tuple_size<decay_t<BoundArgTuple>>::value>()
       )
     )
{
  using Indices = agency::detail::make_index_sequence<std::tuple_size<decay_t<BoundArgTuple>>::value>;
  return substitute_impl(std::forward<ArgTuple>(arg_tuple), std::forward<BoundArgTuple>(bound_arg_tuple), Indices());
}


template<class F, class... BoundArgs>
class bind_expression
{
  public:
    __host__ __device__
    bind_expression(const F& f, const BoundArgs&... bound_args)
      : fun_(f),
        bound_args_(bound_args...)
    {}

    template<class... OtherArgs>
    __host__ __device__
    auto operator()(OtherArgs&&... args) const
      -> decltype(
           apply(
             *std::declval<const F*>(),
             substitute(
               detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
               *std::declval<const tuple<BoundArgs...>*>()
             )
           )
         )
    {
      return apply(
        fun_,
        substitute(
          detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
          bound_args_
        )
      );
    }

    template<class... OtherArgs>
    __host__ __device__
    auto operator()(OtherArgs&&... args)
      -> decltype(
           apply(
             *std::declval<F*>(),
             substitute(
               detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
               *std::declval<tuple<BoundArgs...>*>()
             )
           )
         )
    {
      return apply(
        fun_,
        substitute(
          detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
          bound_args_
        )
      );
    }

  private:
    F fun_;
    tuple<BoundArgs...> bound_args_;
};


} // end bind_detail


template<class F, class... BoundArgs>
__host__ __device__
detail::bind_detail::bind_expression<
  detail::bind_detail::decay_t<F>,
  detail::bind_detail::decay_t<BoundArgs>...
> bind(F&& f, BoundArgs&&... bound_args)
{
  using namespace bind_detail;
  return bind_expression<decay_t<F>,decay_t<BoundArgs>...>(std::forward<F>(f), std::forward<BoundArgs>(bound_args)...);
}


} // end detail
} // end cuda

