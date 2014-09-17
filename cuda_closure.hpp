#pragma once

#include <utility>
#include <thrust/tuple.h>
#include <integer_sequence>


template<typename F, typename Arg1, typename Tuple, size_t... I>
__host__ __device__
auto __apply_impl(F&& f, Arg1&& arg1, Tuple&& t, std::index_sequence<I...>)
  -> decltype(
       std::forward<F>(f)(
         std::forward<Arg1>(arg1),
         thrust::get<I>(std::forward<Tuple>(t))...
       )
     )
{
  return std::forward<F>(f)(
    std::forward<Arg1>(arg1),
    thrust::get<I>(std::forward<Tuple>(t))...
  );
}


template<typename F, typename Arg1, typename Tuple>
__host__ __device__
auto __apply(F&& f, Arg1&& arg1, Tuple&& t)
  -> decltype(
       __apply_impl(
         std::forward<F>(f),
         std::forward<Arg1>(arg1),
         std::forward<Tuple>(t),
         std::make_index_sequence<thrust::tuple_size<std::decay_t<Tuple>>::value>()
       )
     )
{
  using Indices = std::make_index_sequence<thrust::tuple_size<std::decay_t<Tuple>>::value>;
  return __apply_impl(
    std::forward<F>(f),
    std::forward<Arg1>(arg1),
    std::forward<Tuple>(t),
    Indices()
  );
}


template<class Function, class... Args>
class cuda_closure
{
  public:
    __host__ __device__
    cuda_closure(const Function& f, const Args&... args)
      : f(f),
        args(args...)
    {}

    __host__ __device__
    void operator()()
    {
      __apply(f, args);
    }

    template<class Arg1>
    __host__ __device__
    void operator()(Arg1&& arg1)
    {
      __apply(f, std::forward<Arg1>(arg1), args);
    }

  private:
    Function f;
    thrust::tuple<Args...> args;
};


template<class Function, class... Args>
__host__ __device__
cuda_closure<Function,Args...> make_cuda_closure(const Function& f, const Args&... args)
{
  return cuda_closure<Function,Args...>(f, args...);
}

