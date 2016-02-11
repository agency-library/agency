#pragma once

#include <agency/detail/config.hpp>
#include <agency/future.hpp>
#include <agency/executor_traits.hpp>
#include <agency/detail/executor_traits/container_factory.hpp>
#include <type_traits>
#include <utility>

namespace agency
{
namespace detail
{
namespace executor_traits_detail
{


template<class Executor, class Future>
struct has_single_future_share_future
{
  template<class Executor1,
           class = decltype(std::declval<Executor1>().share_future(*std::declval<Future*>()))
          >
  static std::true_type test(int);

  template<class Executor1>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<Executor>(0))::value;
};


template<class Executor, class Future>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_single_future_share_future<Executor,Future>::value,
  typename executor_traits<Executor>::template shared_future<typename future_traits<Future>::value_type>
>::type
  single_future_share_future(Executor& exec, Future& fut)
{
  return exec.share_future(fut);
}


template<class Executor, class Future>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_single_future_share_future<Executor,Future>::value,
  typename executor_traits<Executor>::template shared_future<typename future_traits<Future>::value_type>
>::type
  single_future_share_future(Executor&, Future& fut)
{
  return future_traits<Future>::share(fut);
}


template<class Executor, class Future, class Factory>
struct has_multi_future_share_future_with_factory
{
  using shape_type = typename executor_traits<Executor>::shape_type;

  template<class Executor1,
           class = decltype(std::declval<Executor1>().share_future(*std::declval<Future*>(), std::declval<Factory>(), std::declval<shape_type>()))
           >
  static std::true_type test(int);

  template<class Executor1>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<Executor>(0))::value;
};


__agency_hd_warning_disable__
template<class Executor, class Future, class Factory>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_multi_future_share_future_with_factory<Executor,Future,Factory>::value,
  typename std::result_of<Factory(typename executor_traits<Executor>::shape_type)>::type
>::type
  multi_future_share_future_with_factory(Executor& ex, Future& fut, Factory result_factory, typename executor_traits<Executor>::shape_type shape)
{
  // XXX seems like this would be the superior implementation but I'm afraid of data races to ex
  //auto share_me = executor_traits::share(ex, fut);
  //return executor_traits::execute(ex, result_factory, shape, [=,&ex](auto)
  //{
  //  return executor_traits<Executor>::share_future(ex, share_me);
  //});

  auto results = result_factory(shape);

  auto share_me = executor_traits<Executor>::share_future(ex, fut);

  for(auto& sf : results)
  {
    sf = executor_traits<Executor>::share_future(ex, share_me);
  }

  return std::move(results);
} // end multi_future_share_future_with_factory()


template<class Executor, class Future, class Factory>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_multi_future_share_future_with_factory<Executor,Future,Factory>::value,
  typename std::result_of<Factory(typename executor_traits<Executor>::shape_type)>::type
>::type
  multi_future_share_future_with_factory(Executor& ex, Future& fut, Factory result_factory, typename executor_traits<Executor>::shape_type shape)
{
  return ex.share_future(fut, result_factory, shape);
} // end multi_future_share_future_with_factory()


template<class Executor, class Future>
struct has_multi_future_share_future
{
  using shape_type = typename executor_traits<Executor>::shape_type;

  template<class Executor1,
           class = decltype(std::declval<Executor1>().share_future(*std::declval<Future*>(), std::declval<shape_type>()))
           >
  static std::true_type test(int);

  template<class Executor1>
  static std::false_type test(...);

  static constexpr bool value = decltype(test<Executor>(0))::value;
};


template<class Executor, class Future>
__AGENCY_ANNOTATION
typename std::enable_if<
  !has_multi_future_share_future<Executor,Future>::value,
  typename executor_traits<Executor>::template container<
    typename executor_traits<Executor>::template shared_future<
      typename future_traits<Future>::value_type
    >
  >
>::type
  multi_future_share_future(Executor& ex, Future& fut, typename executor_traits<Executor>::shape_type shape)
{
  using container_type = typename executor_traits<Executor>::template container<
    typename executor_traits<Executor>::template shared_future<
      typename future_traits<Future>::value_type
    >
  >;

  return executor_traits<Executor>::share_future(ex, fut, container_factory<container_type>{}, shape);
} // end multi_future_share_future()


template<class Executor, class Future>
__AGENCY_ANNOTATION
typename std::enable_if<
  has_multi_future_share_future<Executor,Future>::value,
  typename executor_traits<Executor>::template container<
    typename executor_traits<Executor>::template shared_future<
      typename future_traits<Future>::value_type
    >
  >
>::type
  multi_future_share_future(Executor& ex, Future& fut, typename executor_traits<Executor>::shape_type shape)
{
  return ex.share_future(fut, shape);
} // end multi_future_share_future()


} // end executor_traits_detail
} // end detail


template<class Executor>
  template<class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template shared_future<typename future_traits<Future>::value_type>
  executor_traits<Executor>
    ::share_future(typename executor_traits<Executor>::executor_type& ex, Future& fut)
{
  return detail::executor_traits_detail::single_future_share_future(ex, fut);
} // end share_future()


template<class Executor>
  template<class Future, class Factory>
__AGENCY_ANNOTATION
typename std::result_of<Factory(typename executor_traits<Executor>::shape_type)>::type
  executor_traits<Executor>
    ::share_future(typename executor_traits<Executor>::executor_type& ex, Future& fut, Factory result_factory, typename executor_traits<Executor>::shape_type shape)
{
  return detail::executor_traits_detail::multi_future_share_future_with_factory(ex, fut, result_factory, shape);
} // end share_future()


template<class Executor>
  template<class Future>
__AGENCY_ANNOTATION
typename executor_traits<Executor>::template container<
  typename executor_traits<Executor>::template shared_future<
    typename future_traits<Future>::value_type
  >
>
  executor_traits<Executor>
    ::share_future(typename executor_traits<Executor>::executor_type& ex, Future& fut, typename executor_traits<Executor>::shape_type shape)
{
  return detail::executor_traits_detail::multi_future_share_future(ex, fut, shape);
} // end share_future()


} // end agency

