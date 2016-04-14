#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/integer_sequence.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple.hpp>
#include <functional>
#include <type_traits>
#include <utility>
#include <tuple>

namespace agency
{
namespace detail
{


namespace bind_detail
{


template<typename F, typename Tuple, size_t... I>
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
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
__AGENCY_ANNOTATION
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
    __AGENCY_ANNOTATION
    bind_expression(const F& f, const BoundArgs&... bound_args)
      : fun_(f),
        bound_args_(bound_args...)
    {}

    template<class... OtherArgs>
    __AGENCY_ANNOTATION
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
    __AGENCY_ANNOTATION
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


template<class F, class... BoundArgs>
__AGENCY_ANNOTATION
bind_detail::bind_expression<decay_t<F>, decay_t<BoundArgs>...>
  bind(F&& f, BoundArgs&&... bound_args)
{
  using namespace bind_detail;
  return bind_expression<decay_t<F>,decay_t<BoundArgs>...>(std::forward<F>(f), std::forward<BoundArgs>(bound_args)...);
}


template<int I>
struct placeholder {};


namespace placeholders
{


constexpr placeholder<0>   _1{};
constexpr placeholder<1>   _2{};
constexpr placeholder<2>   _3{};
constexpr placeholder<3>   _4{};
constexpr placeholder<4>   _5{};
constexpr placeholder<5>   _6{};
constexpr placeholder<6>   _7{};
constexpr placeholder<7>   _8{};
constexpr placeholder<8>   _9{};
constexpr placeholder<9>   _10{};


} // end placeholders
} // end detail
} // end agency


namespace std
{


// XXX not sure we require this specialization since we don't actually use std::bind() for anything
template<int I>
struct is_placeholder<agency::detail::placeholder<I>> : std::integral_constant<int,I+1> {};


} // end std

