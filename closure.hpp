#pragma once

#include <tuple>
#include <type_traits>


template<typename Function, typename... Args>
struct closure
{
  mutable Function f;
  std::tuple<Args...> args;

  closure(Function f, Args... args)
    : f(f),
      args(args...)
  {}

  typename std::result_of<
    Function(Args...)
  >::type
    operator()() const
  {
    return std::apply(f, args);
  }
};


template<class Function, class... Args>
auto make_closure(Function&& f, Args&&... args) ->
  closure<Function, Args...>
{
  return closure<Function,Args...>(std::forward<Function>(f), std::forward<Args>(args)...);
}

