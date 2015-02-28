#pragma once

#include <type_traits>
#include <utility>
#include <agency/detail/config.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/cuda/detail/tuple.hpp>
#include <functional>
#include <thrust/functional.h>


namespace agency
{
namespace cuda
{
namespace detail
{
namespace bind_detail
{


template<class T>
using decay_t = agency::detail::decay_t<T>;


template<typename F, typename Tuple, size_t... I>
__host__ __device__
auto apply_impl(F&& f, Tuple&& t, agency::detail::index_sequence<I...>)
  -> decltype(
       std::forward<F>(f)(
         detail::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return std::forward<F>(f)(
    detail::get<I>(std::forward<Tuple>(t))...
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


template<class ArgTuple, class BoundArg,
         class = typename std::enable_if<
           (std::is_placeholder<decay_t<BoundArg>>::value == 0)
         >::type>
__host__ __device__
auto substitute_arg(ArgTuple&&, BoundArg&& bound_arg)
  -> decltype(
       std::forward<BoundArg>(bound_arg)
     )
{
  return std::forward<BoundArg>(bound_arg);
}




template<class ArgTuple, class BoundArg,
         class = typename std::enable_if<
           (std::is_placeholder<BoundArg>::value > 0)
         >::type>
__host__ __device__
auto substitute_arg(ArgTuple&& arg_tuple, const BoundArg&)
  -> decltype(
       detail::get<
         static_cast<size_t>(std::is_placeholder<BoundArg>::value) - 1
       >(std::forward<ArgTuple>(arg_tuple))
     )
{
  return detail::get<
    static_cast<size_t>(std::is_placeholder<BoundArg>::value) - 1
  >(std::forward<ArgTuple>(arg_tuple));
}


template<class ArgTuple, class BoundArgTuple, size_t... I>
__host__ __device__
auto substitute_impl(ArgTuple&& arg_tuple, BoundArgTuple&& bound_arg_tuple, agency::detail::index_sequence<I...>)
  -> decltype(
       detail::forward_as_tuple(
         substitute_arg(
           std::forward<ArgTuple>(arg_tuple),
           detail::get<I>(std::forward<BoundArgTuple>(bound_arg_tuple))
         )...
       )
     )
{
  return detail::forward_as_tuple(
    substitute_arg(
      std::forward<ArgTuple>(arg_tuple),
      detail::get<I>(std::forward<BoundArgTuple>(bound_arg_tuple))
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
  private:
    F fun_;
    tuple<BoundArgs...> bound_args_;

  public:
    __host__ __device__
    bind_expression(const F& f, const BoundArgs&... bound_args)
      : fun_(f),
        bound_args_(bound_args...)
    {}

    template<class... OtherArgs>
    __host__ __device__
    auto operator()(OtherArgs&&... args)
      -> decltype(
           apply(
             fun_,
             substitute(
               detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
               bound_args_
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
    auto operator()(OtherArgs&&... args) const
      -> decltype(
           apply(
             fun_,
             substitute(
               detail::forward_as_tuple(std::forward<OtherArgs>(args)...),
               bound_args_
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
};


} // end bind_detail


// XXX use thrust's placeholders instead of agency's
//     because unlike thrust's placeholders, agency's
//     placeholders are undefined in __device__ code
namespace placeholders = thrust::placeholders;


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
} // end agency


namespace std
{


template<unsigned int I>
struct is_placeholder<
  thrust::detail::functional::actor<
    thrust::detail::functional::argument<I>
  >
> : std::integral_constant<int, I+1>
{};


} // end std

