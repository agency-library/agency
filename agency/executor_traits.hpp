#pragma once

#include <agency/future.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/bind.hpp>
#include <agency/execution_categories.hpp>
#include <agency/detail/tuple.hpp>
#include <agency/detail/shape_cast.hpp>


namespace agency
{
namespace detail
{


__DEFINE_HAS_NESTED_TYPE(has_index_type, index_type);
__DEFINE_HAS_NESTED_TYPE(has_shape_type, shape_type);
__DEFINE_HAS_NESTED_TYPE(has_execution_category, execution_category);
__DEFINE_HAS_NESTED_CLASS_TEMPLATE(has_future_template, future);


template<class T>
struct nested_index_type
{
  using type = typename T::index_type;
};


template<class T, class Default = size_t>
struct nested_index_type_with_default
  : agency::detail::lazy_conditional<
      agency::detail::has_index_type<T>::value,
      nested_index_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct nested_shape_type
{
  using type = typename T::shape_type;
};


template<class T, class Default = size_t>
struct nested_shape_type_with_default
  : agency::detail::lazy_conditional<
      agency::detail::has_shape_type<T>::value,
      nested_shape_type<T>,
      agency::detail::identity<Default>
    >
{};


template<class T>
struct nested_future_template
{
  template<class U>
  using type_template = typename T::template future<U>;
};


template<bool condition, class Then, class Else>
struct lazy_conditional_template
{
  template<class T>
  using type_template = typename Then::template type_template<T>;
};

template<class Then, class Else>
struct lazy_conditional_template<false, Then, Else>
{
  template<class T>
  using type_template = typename Else::template type_template<T>;
};


template<template<class> class T>
struct identity_template
{
  template<class U>
  using type_template = T<U>;
};


template<class T, template<class> class Default = std::future>
struct nested_future_with_default
  : agency::detail::lazy_conditional_template<
      agency::detail::has_future_template<T,void>::value,
      nested_future_template<T>,
      identity_template<Default>
    >
{};


template<class Executor, class T, class TypeList>
struct has_then_execute_impl;


template<class Executor, class T, class... Types>
struct has_then_execute_impl<Executor, T, type_list<Types...>>
{
  using index_type = typename nested_index_type_with_default<
    Executor
  >::type;

  using shape_type = typename nested_shape_type_with_default<
    Executor
  >::type;

  template<class U>
  using future = typename nested_future_with_default<
    Executor
  >::template type_template<U>;

  struct dummy_functor
  {
    void operator()(const index_type&, T&, Types&...) {}

    // overload for void past parameter
    void operator()(const index_type&, Types&...) {}
  };

  template<class Executor1,
           class = decltype(
             std::declval<Executor1>().then_execute(
               *std::declval<future<T>*>(),
               std::declval<dummy_functor>(),
               std::declval<shape_type>(),
               *std::declval<Types*>()...
             )
           )>
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class T, bool = has_execution_category<T>::value>
struct has_then_execute : std::false_type {};

template<class T>
struct has_then_execute<T,true> 
  : has_then_execute_impl<
      T,
      int,
      repeat_type<
        int, execution_depth<typename T::execution_category>::value
      >
    >::type
{};


template<class Executor, class T, class... Args>
struct has_make_ready_future_impl
{
  template<
    class Executor2,
    typename = decltype(
      std::declval<Executor2*>()->template make_ready_future<T>(
        std::declval<Args>()...
      )
    )
  >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<Executor>(0));
};


template<class Executor, class T, class... Args>
using has_make_ready_future = typename has_make_ready_future_impl<Executor,T,Args...>::type;


} // end detail


template<class T>
struct is_executor
  : std::integral_constant<
      bool,
      detail::has_execution_category<T>::value &&
      detail::has_then_execute<T>::value
    >
{};


template<class Executor>
struct executor_traits
{
  private:
    template<class T>
    struct executor_index
    {
      using type = typename T::index_type;
    };

    template<class T>
    struct executor_shape
    {
      using type = typename T::shape_type;
    };

  public:
    using executor_type = Executor;

    using execution_category = typename Executor::execution_category;

    using index_type = typename detail::nested_index_type_with_default<
      executor_type,
      size_t
    >::type;

    using shape_type = typename detail::lazy_conditional<
      detail::has_shape_type<executor_type>::value,
      executor_shape<executor_type>,
      detail::identity<index_type>
    >::type;

  private:
    template<class T, class U>
    struct has_future_impl
    {
      template<class> static std::false_type test(...);
      template<class X> static std::true_type  test(typename X::template future<U>* = 0);

      using type = decltype(test<T>(0));
    };
    
    template<class T, class U>
    struct has_future : has_future_impl<T,U>::type {};

    template<class T, class U, bool = has_future<T,U>::value>
    struct executor_future
    {
      using type = typename T::template future<U>;
    };

    template<class T, class U>
    struct executor_future<T,U,false>
    {
      using type = std::future<U>;
    };

  public:
    template<class T>
    using future = typename executor_future<executor_type,T>::type;

  private:
    static future<void> make_ready_future_impl(executor_type& ex, std::true_type)
    {
      return ex.make_ready_future();
    }

    static future<void> make_ready_future_impl(executor_type&, std::false_type)
    {
      return future_traits<future<void>>::make_ready();
    }

    template<class T, class... Args>
    static future<T> make_ready_future_impl(std::true_type, executor_type& ex, Args&&... args)
    {
      return ex.template make_ready_future<T>(std::forward<Args>(args)...);
    }

    template<class T, class... Args>
    static future<T> make_ready_future_impl(std::false_type, executor_type&, Args&&... args)
    {
      return future_traits<future<T>>::template make_ready<T>(std::forward<Args>(args)...);
    }

    template<class T>
    struct is_future : detail::is_instance_of_future<T,future> {};

  public:
    template<class T, class... Args>
    static future<T> make_ready_future(executor_type& ex, Args&&... args)
    {
      return make_ready_future_impl<T>(detail::has_make_ready_future<executor_type,T,Args&&...>(), ex, std::forward<Args>(args)...);
    }


  private:
    template<class Executor1, class T, class Future>
    struct has_future_cast_impl
    {
      template<
        class Executor2,
        typename = decltype(std::declval<Executor2*>()->template future_cast<T>(std::declval<Future>()))
      >
      static std::true_type test(int);
    
      template<class>
      static std::false_type test(...);
    
      using type = decltype(test<Executor1>(0));
    };

    template<class T, class Future>
    using has_future_cast = typename has_future_cast_impl<executor_type,T,Future>::type;

    template<class T, class Future>
    static future<T> future_cast_impl2(executor_type& ex, Future& from, std::true_type)
    {
      return ex.template future_cast<T>(from);
    }

    template<class T, class Future>
    static future<T> future_cast_impl2(executor_type& ex, Future& from, std::false_type)
    {
      using value_type = typename future_traits<Future>::value_type;

      return executor_traits::then_execute(ex, from, [](index_type, value_type& from_value)
      {
        return static_cast<T>(std::move(from_value));
      },
      detail::shape_cast<shape_type>(1)
      );
    }

    template<class Future>
    struct has_discard_value_impl
    {
      template<class Future1,
               typename = decltype(future_traits<Future1>::discard_value())
              >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<Future>(0));
    };

    template<class Future>
    using has_discard_value = typename has_discard_value_impl<Future>::type;

    // check for cheap cast to void
    template<class T, class Future,
             class = typename std::enable_if<
               is_future<Future>::value && std::is_void<T>::value
             >::type,
             class = typename std::enable_if<
               has_discard_value<Future>::value
             >::type
            >
    static future<T> future_cast_impl1(executor_type&, Future& from)
    {
      // we don't need to involve the executor at all
      return future_traits<Future>::discard_value(from);
    }

    template<class T, class Future,
             class = typename std::enable_if<
               !is_future<Future>::value || !std::is_void<T>::value || !has_discard_value<Future>::value
             >::type
            >
    static future<T> future_cast_impl1(executor_type& ex, Future& from)
    {
      return future_cast_impl2<T>(ex, from, has_future_cast<T,Future>());
    }

  public:
    template<class T, class Future>
    static future<T> future_cast(executor_type& ex, Future& from)
    {
      return future_cast_impl1<T>(ex, from);
    }

    // XXX generalize this to interoperate with other Futures
    // XXX we can use async_execute & call fut.get() when depending on foreign Futures
    template<class Future, class Function, class T1, class... Types,
             class = typename std::enable_if<
               is_future<Future>::value
             >::type>
    static future<void> then_execute(executor_type& ex, Future& fut, Function f, shape_type shape, T1&& outer_shared_init, Types&&... inner_shared_inits)
    {
      return ex.then_execute(fut, f, shape, std::forward<T1>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
    }

  private:
    template<class T, class Function>
    struct test_for_then_execute_without_shared_inits
    {
      template<
        class Executor2,
        typename = decltype(std::declval<Executor2*>()->then_execute(
        *std::declval<future<T>*>(),
        std::declval<Function>(),
        std::declval<shape_type>()))
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<executor_type>(0));
    };

    template<class T, class Function>
    using has_then_execute_without_shared_inits = typename test_for_then_execute_without_shared_inits<T,Function>::type;

    template<class T, class Function>
    static future<void> then_execute_without_shared_inits_impl(executor_type& ex, future<T>& fut, Function f, shape_type shape, std::true_type)
    {
      return ex.then_execute(fut, f, shape);
    }

    template<class T, class Function, class... Types, size_t... Indices,
             class = typename std::enable_if<
               std::is_void<T>::value
             >::type>
    static future<void> then_execute_without_shared_inits_impl(executor_type& ex, future<void>& fut, Function f, shape_type shape, const detail::tuple<Types...>& dummy_shared_inits, detail::index_sequence<Indices...>)
    {
      return executor_traits::then_execute(ex, fut, [=](index_type index, Types&...) mutable
      {
        f(index);
      },
      shape,
      detail::get<Indices>(dummy_shared_inits)...
      );
    }

    template<class T, class Function, class... Types, size_t... Indices,
             class = typename std::enable_if<
               !std::is_void<T>::value
             >::type>
    static future<void> then_execute_without_shared_inits_impl(executor_type& ex, future<T>& fut, Function f, shape_type shape, const detail::tuple<Types...>& dummy_shared_inits, detail::index_sequence<Indices...>)
    {
      return executor_traits::then_execute(ex, fut, [=](index_type index, T& past_parameter, const Types&...) mutable
      {
        f(index, past_parameter);
      },
      shape,
      detail::get<Indices>(dummy_shared_inits)...
      );
    }

    template<class T, class Function>
    static future<void> then_execute_without_shared_inits_impl(executor_type& ex, future<T>& fut, Function f, shape_type shape, std::false_type)
    {
      constexpr size_t depth = detail::execution_depth<execution_category>::value;

      // create dummy shared initializers
      auto dummy_tuple = detail::tuple_repeat<depth>(detail::ignore);

      return executor_traits::then_execute_without_shared_inits_impl<T>(ex, fut, f, shape, dummy_tuple, detail::make_index_sequence<depth>());
    }

  public:
    template<class Future, class Function,
             class = typename std::enable_if<
               is_future<Future>::value
             >::type>
    static future<void> then_execute(executor_type& ex, Future& fut, Function f, shape_type shape)
    {
      using value_type = typename future_traits<Future>::value_type;
      return executor_traits::then_execute_without_shared_inits_impl<value_type>(ex, fut, f, shape, has_then_execute_without_shared_inits<value_type,Function>());
    }

  private:
    template<class Function, class... T>
    struct test_for_async_execute_with_shared_inits
    {
      template<
        class Executor2,
        typename = decltype(
          std::declval<Executor2*>()->async_execute(
            std::declval<Function>(),
            std::declval<shape_type>(),
            std::declval<T>()...
          )
        )
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<executor_type>(0));
    };

    template<class Function, class... T>
    using has_async_execute_with_shared_inits = typename test_for_async_execute_with_shared_inits<Function,T...>::type;

    template<class Function, class... T>
    static future<void> async_execute_with_shared_inits_impl(std::true_type, executor_type& ex, Function f, shape_type shape, T&&... shared_inits)
    {
      return ex.async_execute(f, shape, std::forward<T>(shared_inits)...);
    }

    template<class Function, class... T>
    static future<void> async_execute_with_shared_inits_impl(std::false_type, executor_type& ex, Function f, shape_type shape, T&&... shared_inits)
    {
      auto ready = executor_traits::template make_ready_future<void>(ex);
      return executor_traits::then_execute(ex, ready, f, shape, std::forward<T>(shared_inits)...);
    }

  public:
    template<class Function, class T, class... Types>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape, T&& outer_shared_init, Types&&... inner_shared_inits)
    {
      return executor_traits::async_execute_with_shared_inits_impl(has_execute_with_shared_inits<Function,T,Types...>(), ex, f, shape, std::forward<T>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
    }

  private:
    template<class Function>
    struct test_for_async_execute_without_shared_inits
    {
      template<
        class Executor2,
        typename = decltype(std::declval<Executor2*>()->async_execute(
        std::declval<Function>(),
        std::declval<shape_type>()))
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<executor_type>(0));
    };

    template<class Function>
    using has_async_execute_without_shared_inits = typename test_for_async_execute_without_shared_inits<Function>::type;

    template<class Function>
    static future<void> async_execute_without_shared_inits_impl(executor_type& ex, Function f, shape_type shape, std::true_type)
    {
      return ex.async_execute(f, shape);
    }

    template<class Function, class... Types, size_t... Indices>
    static future<void> async_execute_without_shared_inits_impl(executor_type& ex, Function f, shape_type shape, const detail::tuple<Types...>& dummy_shared_inits, detail::index_sequence<Indices...>)
    {
      return executor_traits::async_execute(ex, [=](index_type index, Types&...) mutable
      {
        f(index);
      },
      shape,
      detail::get<Indices>(dummy_shared_inits)...
      );
    }

    template<class Function>
    static future<void> async_execute_without_shared_inits_impl(executor_type& ex, Function f, shape_type shape, std::false_type)
    {
      constexpr size_t depth = detail::execution_depth<execution_category>::value;

      // create dummy shared initializers
      auto dummy_tuple = detail::tuple_repeat<depth>(detail::ignore);

      return executor_traits::async_execute_without_shared_inits_impl(ex, f, shape, dummy_tuple, detail::make_index_sequence<depth>());
    }

  public:
    template<class Function>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape)
    {
      return executor_traits::async_execute_without_shared_inits_impl(ex, f, shape, has_async_execute_without_shared_inits<Function>());
    }

  private:
    template<class Function, class... T>
    struct test_for_execute_with_shared_inits
    {
      template<
        class Executor2,
        typename = decltype(
          std::declval<Executor2*>()->execute(
            std::declval<Function>(),
            std::declval<shape_type>(),
            std::declval<T>()...
          )
        )
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<executor_type>(0));
    };

    template<class Function, class... T>
    using has_execute_with_shared_inits = typename test_for_execute_with_shared_inits<Function,T...>::type;

    template<class Function, class... T>
    static void execute_with_shared_inits_impl(std::true_type, executor_type& ex, Function f, shape_type shape, T&&... shared_inits)
    {
      ex.execute(f, shape, std::forward<T>(shared_inits)...);
    }

    template<class Function, class... T>
    static void execute_with_shared_inits_impl(std::false_type, executor_type& ex, Function f, shape_type shape, T&&... shared_inits)
    {
      executor_traits::async_execute(ex, f, shape, std::forward<T>(shared_inits)...).wait();
    }

  public:
    template<class Function, class T, class... Types>
    static void execute(executor_type& ex, Function f, shape_type shape, T&& outer_shared_init, Types&&... inner_shared_inits)
    {
      executor_traits::execute_with_shared_inits_impl(has_execute_with_shared_inits<Function,T,Types...>(), ex, f, shape, std::forward<T>(outer_shared_init), std::forward<Types>(inner_shared_inits)...);
    }

  private:
    template<class Function>
    struct test_for_execute
    {
      template<
        class Executor2,
        class Function2,
        typename = decltype(std::declval<Executor2*>()->execute(
        std::declval<Function2>(),
        std::declval<shape_type>()))
      >
      static std::true_type test(int);

      template<class,class>
      static std::false_type test(...);

      using type = decltype(test<executor_type,Function>(0));
    };

    template<class Function>
    using has_execute = typename test_for_execute<Function>::type;

    template<class Function>
    static void execute_impl(executor_type& ex, Function f, shape_type shape, std::true_type)
    {
      ex.execute(f, shape);
    }

    template<class Function>
    static void execute_impl(executor_type& ex, Function f, shape_type shape, std::false_type)
    {
      executor_traits::async_execute(ex, f, shape).wait();
    }

  public:
    template<class Function>
    static void execute(executor_type& ex, Function f, shape_type shape)
    {
      executor_traits::execute_impl(ex, f, shape, has_execute<Function>());
    }
};


namespace detail
{


template<class Executor, class Enable = void>
struct executor_index {};

template<class Executor>
struct executor_index<Executor, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::index_type;
};

template<class Executor>
using executor_index_t = typename executor_index<Executor>::type;


template<class Executor, class Enable = void>
struct executor_shape {};

template<class Executor>
struct executor_shape<Executor, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::shape_type;
};

template<class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


template<class Executor, class T, class Enable = void>
struct executor_future {};

template<class Executor, class T>
struct executor_future<Executor, T, typename std::enable_if<is_executor<Executor>::value>::type>
{
  using type = typename executor_traits<Executor>::template future<T>;
};

template<class Executor, class T>
using executor_future_t = typename executor_future<Executor,T>::type;


} // end detail


} // end agency

