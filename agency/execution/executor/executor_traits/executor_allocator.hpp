#pragma once

#include <agency/detail/config.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/execution/executor/executor_traits/is_executor.hpp>
#include <agency/execution/executor/executor_traits/detail/member_allocator_or.hpp>
#include <memory>


namespace agency
{
namespace detail
{


template<class Executor, class T, bool Enable = is_executor<Executor>::value>
struct executor_allocator_impl
{
};

template<class Executor, class T>
struct executor_allocator_impl<Executor,T,true>
{
  using type = member_allocator_or_t<Executor,T,std::allocator>;
};


} // end detail


template<class Executor, class T>
struct executor_allocator : detail::executor_allocator_impl<Executor,T> {};

template<class Executor, class T>
using executor_allocator_t = typename executor_allocator<Executor,T>::type;


} // end agency

