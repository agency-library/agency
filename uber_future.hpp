#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/deferred_future.hpp>
#include <agency/cuda/future.hpp>
#include <agency/detail/variant.hpp>

template<class T>
class uber_future
{
  private:
    using variant_type = agency::detail::variant<agency::cuda::deferred_future<T>, agency::cuda::future<T>>;

  public:
    __AGENCY_ANNOTATION
    uber_future() = default;

    __AGENCY_ANNOTATION
    uber_future(uber_future&&) = default;

    template<class Future,
             class = typename std::enable_if<
               std::is_constructible<variant_type,Future&&>::value
             >::type>
    __AGENCY_ANNOTATION
    uber_future(Future&& other)
      : variant_(std::forward<Future>(other))
    {}

    __AGENCY_ANNOTATION
    uber_future& operator=(uber_future&& other) = default;

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
      return agency::detail::visit(visitor, variant_);
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
      return agency::detail::visit(visitor, variant_);
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
      return agency::detail::visit(visitor, variant_);
    }

    template<class... Args,
             class = typename std::enable_if<
               agency::cuda::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    __AGENCY_ANNOTATION
    static uber_future make_ready(Args&&... args)
    {
      return agency::cuda::deferred_future<T>::make_ready(std::forward<Args>(args)...);
    }

  private:
    template<class Function>
    struct then_visitor
    {
      mutable Function f;

      template<class Future>
      __AGENCY_ANNOTATION
      uber_future<
        agency::detail::result_of_continuation_t<
          Function, 
          Future
        >
      >
        operator()(Future& fut) const
      {
        return fut.then(std::move(f));
      }

      template<class Function1>
      static __AGENCY_ANNOTATION
      uber_future<
        agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::future<T>
        >
      >
        async_future_impl(agency::cuda::future<T>& fut, Function1& f,
                          typename std::enable_if<
                            !std::is_move_constructible<Function1>::value
                          >::type* = 0)
      {
        return fut.then(f);
      }

      template<class Function1>
      static __AGENCY_ANNOTATION
      uber_future<
        agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::future<T>
        >
      >
        async_future_impl(agency::cuda::future<T>& fut, Function1& f,
                          typename std::enable_if<
                            std::is_move_constructible<Function1>::value
                          >::type* = 0)
      {
        //return fut.then(std::move(f));

        // XXX Function is often move-only
        //     the problem is that the above fut.then() will result in a CUDA kernel launch
        //     and kernel parameters must be copyable
        //     we should either implement movable CUDA kernel parameters
        //     or find a way to attach a deferred continuation onto an asynchronous CUDA future
        //     there ought to be a way to do it by implementing a deferred_continuation which waits on fut
        // XXX when Function is copyable, we ought to just use fut.then()
        printf("uber_future::then_visitor::operator()(cuda::future): unimplemented\n");
        assert(0);

        using result_type = agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::future<T>
        >;

        return uber_future<result_type>();
      }

      __AGENCY_ANNOTATION
      uber_future<
        agency::detail::result_of_continuation_t<
          Function, 
          agency::cuda::future<T>
        >
      >
        operator()(agency::cuda::future<T>& fut) const
      {
        return async_future_impl(fut, f);
      }
    };

  public:
    template<class Function>
    __AGENCY_ANNOTATION
    uber_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        uber_future
      >
    >
      then(Function&& f)
    {
      auto visitor = then_visitor<typename std::decay<Function>::type>{std::forward<Function>(f)};
      return agency::detail::visit(visitor, variant_);
    }

  private:
    variant_type variant_;
};

