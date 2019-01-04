#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <agency/future/always_ready_future.hpp>
#include <agency/experimental/variant.hpp>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/invoke.hpp>
#include <agency/detail/terminate.hpp>
#include <utility>


namespace agency
{
namespace cuda
{


// forward declaration for future::share()
template<class T> class shared_future;


template<class T>
class future
{
  private:
    template<class U> friend class future;

    using variant_type = agency::experimental::variant<agency::cuda::async_future<T>, agency::always_ready_future<T>>;

  public:
    future() = default;

    future(future&&) = default;

    template<class Future,
             class = typename std::enable_if<
               std::is_constructible<variant_type,Future&&>::value
             >::type>
    __AGENCY_ANNOTATION
    future(Future&& other)
      : variant_(std::forward<Future>(other))
    {
    }

  private:
    template<class U>
    struct converting_move_construct_visitor
    {
      __AGENCY_ANNOTATION
      future operator()(async_future<U>& other) const
      {
        return async_future<T>(std::move(other));
      }
    };

    template<class U>
    __AGENCY_ANNOTATION
    static future converting_move_construct(future<U>&& other)
    {
      return agency::experimental::visit(converting_move_construct_visitor<U>{}, other.variant_);
    }

  public:
    // this constructor allows move construction of our T from a U
    // when that is possible for all the alternative futures
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 async_future<T>,async_future<U>&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    future(future<U>&& other)
      : future(converting_move_construct(std::move(other)))
    {}

    future& operator=(future&& other) = default;

    shared_future<T> share();

    template<class Future>
    __AGENCY_ANNOTATION
    Future& get()
    {
      return agency::experimental::get<Future>(variant_);
    }

    template<class Future>
    __AGENCY_ANNOTATION
    const Future& get() const
    {
      return agency::experimental::get<Future>(variant_);
    }

    template<class Future>
    __AGENCY_ANNOTATION
    Future&& get() &&
    {
      return agency::experimental::get<Future>(std::move(variant_));
    }

    __AGENCY_ANNOTATION
    size_t index() const
    {
      return variant_.index();
    }

  private:
    struct valid_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      bool operator()(const Future& f) const
      {
        return f.valid();
      }
    };

  public:
    __AGENCY_ANNOTATION
    bool valid() const
    {
      auto visitor = valid_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    struct is_ready_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      bool operator()(Future& f) const
      {
        return f.is_ready();
      }
    };

  public:
    __AGENCY_ANNOTATION
    bool is_ready() const
    {
      auto visitor = is_ready_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    struct wait_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      void operator()(Future& f) const
      {
        f.wait();
      }
    };

  public:
    __AGENCY_ANNOTATION
    void wait()
    {
      auto visitor = wait_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    struct get_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      T operator()(Future& f) const
      {
        return f.get();
      }
    };

  public:
    __AGENCY_ANNOTATION
    T get()
    {
      auto visitor = get_visitor();
      return agency::experimental::visit(visitor, variant_);
    }

    template<class... Args,
             class = typename std::enable_if<
               agency::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __AGENCY_ANNOTATION
    static future make_ready(Args&&... args)
    {
      return agency::cuda::async_future<T>::make_ready(std::forward<Args>(args)...);
    }

  private:
    template<class Function>
    struct then_visitor
    {
      mutable Function f;

      template<class Future>
      __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function, 
          Future
        >
      >
        operator()(Future& fut) const
      {
        return fut.then(std::move(f));
      }

      __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function, 
          agency::cuda::async_future<T>
        >
      >
        operator()(agency::cuda::async_future<T>& fut) const
      {
        return fut.then(f);
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        future
      >
    >
      then(Function&& f)
    {
      auto visitor = then_visitor<typename std::decay<Function>::type>{std::forward<Function>(f)};
      return agency::experimental::visit(visitor, variant_);
    }

  private:
    template<class Function>
    struct then_and_leave_valid_visitor
    {
      mutable Function f;

      template<class Future>
      __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function, 
          Future
        >
      >
        operator()(Future& fut) const
      {
        return fut.then_and_leave_valid(std::move(f));
      }

      template<class Function1>
      static __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::async_future<T>
        >
      >
        async_future_impl(agency::cuda::async_future<T>& fut, Function1& f,
                          typename std::enable_if<
                            !std::is_move_constructible<Function1>::value
                          >::type* = 0)
      {
        return fut.then_and_leave_valid(f);
      }

      template<class Function1>
      static __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::async_future<T>
        >
      >
        async_future_impl(agency::cuda::async_future<T>& /* fut */, Function1& /* f */,
                          typename std::enable_if<
                            std::is_move_constructible<Function1>::value
                          >::type* = 0)
      {
        //return fut.then_and_leave_valid(std::move(f));

        // XXX Function is often move-only
        //     the problem is that the above fut.then_and_leave_valid() will result in a CUDA kernel launch
        //     and kernel parameters must be copyable
        //     we should either implement movable CUDA kernel parameters
        //     or find a way to attach a deferred continuation onto an asynchronous CUDA future
        //     there ought to be a way to do it by implementing a deferred_continuation which waits on fut
        // XXX when Function is copyable, we ought to just use fut.then()
        agency::detail::throw_runtime_error("future::then_and_leave_valid_visitor::operator()(cuda::future): unimplemented");

        using result_type = agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::async_future<T>
        >;

        return future<result_type>();
      }

      __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_continuation_t<
          Function, 
          agency::cuda::async_future<T>
        >
      >
        operator()(agency::cuda::async_future<T>& fut) const
      {
        return async_future_impl(fut, f);
      }
    };

    template<class Function>
    __AGENCY_ANNOTATION
    future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        future
      >
    >
      then_and_leave_valid(Function&& f)
    {
      auto visitor = then_and_leave_valid_visitor<typename std::decay<Function>::type>{std::forward<Function>(f)};
      return agency::experimental::visit(visitor, variant_);
    }

    variant_type variant_;

    using get_ref_result_type = typename std::conditional<
      std::is_void<T>::value,
      void,
      typename agency::detail::lazy_add_lvalue_reference<
        agency::detail::identity<T>
      >::type
    >::type;

    struct get_ref_visitor
    {
      template<class Future>
      __AGENCY_ANNOTATION
      get_ref_result_type operator()(Future& fut)
      {
        return fut.get_ref();
      }
    };

    get_ref_result_type get_ref()
    {
      get_ref_visitor visitor;
      return agency::experimental::visit(visitor, variant_);
    }

    friend class shared_future<T>;
};


} // end cuda
} // end agency

