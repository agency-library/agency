#pragma once

#include <agency/detail/config.hpp>

namespace agency
{


template<class Future, class T>
struct future_rebind_value
{
  using type = typename Future::template rebind_value<T>;
};


template<template<class> class Future, class FromType, class ToType>
struct future_rebind_value<Future<FromType>,ToType>
{
  using type = Future<ToType>;
};


template<class Future, class T>
using future_rebind_value_t = typename future_rebind_value<Future,T>::type;


} // end agency

