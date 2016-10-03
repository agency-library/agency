#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/member_future_or.hpp>
#include <future>

namespace agency
{
namespace detail
{


template<class Executor, class T, bool Enable = is_executor<Executor>::value>
struct executor_future_impl
{
};

template<class Executor, class T>
struct executor_future_impl<Executor,T,true>
{
  using type = member_future_or_t<Executor,T,std::future>;
};


} // end detail


template<class Executor, class T>
struct executor_future : detail::executor_future_impl<Executor,T> {};

template<class Executor, class T>
using executor_future_t = typename executor_future<Executor,T>::type;


} // end agency

