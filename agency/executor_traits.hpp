#pragma once

#include <future>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/tuple_of_references.hpp>
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

    // XXX we could make .async_execute(f, shape, shared_arg) optional
    //     the default implementation could create a launcher agent to own the shared arg and wait for the
    //     workers
    template<class Function, class T>
    static future<void> async_execute(executor_type& ex, Function f, shape_type shape, T shared_arg)
    {
      return ex.async_execute(f, shape, shared_arg);
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
    template<class Function, class T>
    struct test_for_execute_with_shared_arg
    {
      template<
        class Executor2,
        typename = decltype(std::declval<Executor2*>()->execute(
        std::declval<Function>(),
        std::declval<shape_type>(),
        std::declval<T>()))
      >
      static std::true_type test(int);

      template<class>
      static std::false_type test(...);

      using type = decltype(test<executor_type>(0));
    };

    template<class Function, class T>
    using has_execute_with_shared_arg = typename test_for_execute_with_shared_arg<Function,T>::type;

    template<class Function, class T>
    static void execute_with_shared_arg_impl(executor_type& ex, Function f, shape_type shape, T shared_arg, std::true_type)
    {
      ex.execute(f, shape, shared_arg);
    }

    template<class Function, class T>
    static void execute_with_shared_arg_impl(executor_type& ex, Function f, shape_type shape, T shared_arg, std::false_type)
    {
      executor_traits::async_execute(ex, f, shape, shared_arg).wait();
    }

  public:
    template<class Function, class T>
    static void execute(executor_type& ex, Function f, shape_type shape, T shared_arg)
    {
      executor_traits::execute_with_shared_arg_impl(ex, f, shape, shared_arg, has_execute_with_shared_arg<Function,T>());
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

  private:
    template<class Executor1, class T1>
    struct test_for_shared_param_type
    {
      template<
        class Executor2,
        class T2
      >
      static std::true_type test(typename Executor2::template shared_param_type<T2>*);

      template<class,class>
      static std::false_type test(...);

      using type = decltype(test<Executor1,T1>(0));
    };

    template<class T>
    using has_shared_param_type = typename test_for_shared_param_type<executor_type,T>::type;

    template<class T>
    struct tuple_of_references_t
    {
      using type = decltype(detail::tuple_of_references(*std::declval<T*>()));
    };

    template<class Executor1, class T>
    struct executor_shared_param_type
    {
      using type = typename Executor1::template shared_param_type<T>;
    };

    // check if executor_type has a declared shared_param_type for T
    // if so, use it
    // else, check if execution_category is nested
    // if so, interpret T as a tuple and the shared param type is a tuple of references to T's elements
    // else, the shared param type is just a reference to T
    template<class T>
    struct shared_param_type_impl
      : detail::lazy_conditional<
          has_shared_param_type<T>::value,
          executor_shared_param_type<executor_type,T>,         
          detail::lazy_conditional<
            detail::is_nested_execution_category<execution_category>::value,
            tuple_of_references_t<T>,
            std::add_lvalue_reference<T>
          >
        >
    {
    };

  public:
    template<class T>
    using shared_param_type = typename shared_param_type_impl<T>::type;
};


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

