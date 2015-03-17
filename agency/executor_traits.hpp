#pragma once

#include <future>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/bind.hpp>
#include <agency/execution_categories.hpp>


namespace agency
{
namespace detail
{


__DEFINE_HAS_NESTED_TYPE(has_index_type, index_type);
__DEFINE_HAS_NESTED_TYPE(has_shape_type, shape_type);


} // end detail


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

    using index_type = typename detail::lazy_conditional<
      detail::has_index_type<executor_type>::value,
      executor_index<executor_type>,
      detail::identity<size_t>
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

    // XXX we could make .async_execute(f, shape, shared_inits...) optional
    //     the default implementation could create a launcher agent to own the shared inits and wait for the
    //     workers
    template<class Function, class T, class... Types>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape, T outer_shared_init, Types... inner_shared_inits)
    {
      return ex.async_execute(f, shape, outer_shared_init, inner_shared_inits...);
    }

  private:
    template<class Function>
    struct test_for_async_execute
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
    using has_async_execute = typename test_for_async_execute<Function>::type;

    template<class Function>
    static future<void> async_execute_impl(executor_type& ex, Function f, shape_type shape, std::true_type)
    {
      return ex.async_execute(f, shape);
    }

    template<class Function>
    static future<void> async_execute_impl(executor_type& ex, Function f, shape_type shape, std::false_type)
    {
      return executor_traits::async_execute(ex, [=](index_type index, const shape_type&) mutable
      {
        f(index);
      },
      shape, shape
      );
    }

  public:
    template<class Function>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape)
    {
      return executor_traits::async_execute_impl(ex, f, shape, has_async_execute<Function>());
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
    static void execute_with_shared_inits_impl(std::true_type, executor_type& ex, Function f, shape_type shape, T... shared_inits)
    {
      ex.execute(f, shape, shared_inits...);
    }

    template<class Function, class... T>
    static void execute_with_shared_inits_impl(std::false_type, executor_type& ex, Function f, shape_type shape, T... shared_inits)
    {
      executor_traits::async_execute(ex, f, shape, shared_inits...).wait();
    }

  public:
    template<class Function, class T, class... Types>
    static void execute(executor_type& ex, Function f, shape_type shape, T outer_shared_init, Types... inner_shared_inits)
    {
      executor_traits::execute_with_shared_inits_impl(has_execute_with_shared_inits<Function,T,Types...>(), ex, f, shape, outer_shared_init, inner_shared_inits...);
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


template<class Executor>
struct executor_index
{
  using type = typename executor_traits<Executor>::index_type;
};


template<class Executor>
using executor_index_t = typename executor_index<Executor>::type;


template<class Executor>
struct executor_shape
{
  using type = typename executor_traits<Executor>::shape_type;
};


template<class Executor>
using executor_shape_t = typename executor_shape<Executor>::type;


template<class Executor, class T>
struct executor_future
{
  using type = typename executor_traits<Executor>::template future<T>;
};

template<class Executor, class T>
using executor_future_t = typename executor_future<Executor,T>::type;


} // end detail


// XXX eliminate this
template<class Executor, class Function, class... Args>
typename executor_traits<Executor>::template future<void>
  bulk_async(Executor& ex,
             typename executor_traits<Executor>::shape_type shape,
             Function&& f,
             Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), detail::placeholders::_1, std::forward<Args>(args)...);
  return executor_traits<Executor>::async_execute(ex, f, shape);
}


// XXX eliminate this
template<class Executor, class Function, class... Args>
void bulk_invoke(Executor& ex,
                 typename executor_traits<Executor>::shape_type shape,
                 Function&& f,
                 Args&&... args)
{
  auto g = detail::bind(std::forward<Function>(f), detail::placeholders::_1, std::forward<Args>(args)...);
  return executor_traits<Executor>::execute(ex, f, shape);
}


} // end agency

