#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/future.hpp>
#include <agency/detail/variant.hpp>

// forward declaration for uber_future::share()
template<class T> class shared_uber_future;


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

    shared_uber_future<T> share();

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
    template<class Function>
    struct then_and_leave_valid_visitor
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
        return fut.then_and_leave_valid(std::move(f));
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
        return fut.then_and_leave_valid(f);
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
        //return fut.then_and_leave_valid(std::move(f));

        // XXX Function is often move-only
        //     the problem is that the above fut.then_and_leave_Valid() will result in a CUDA kernel launch
        //     and kernel parameters must be copyable
        //     we should either implement movable CUDA kernel parameters
        //     or find a way to attach a deferred continuation onto an asynchronous CUDA future
        //     there ought to be a way to do it by implementing a deferred_continuation which waits on fut
        // XXX when Function is copyable, we ought to just use fut.then()
        printf("uber_future::then_and_leave_valid_visitor::operator()(cuda::future): unimplemented\n");
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

    template<class Function>
    __AGENCY_ANNOTATION
    uber_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        uber_future
      >
    >
      then_and_leave_valid(Function&& f)
    {
      auto visitor = then_and_leave_valid_visitor<typename std::decay<Function>::type>{std::forward<Function>(f)};
      return agency::detail::visit(visitor, variant_);
    }


    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    struct bulk_then_visitor
    {
      Function f;
      Factory result_factory;
      Shape shape;
      IndexFunction index_function;
      OuterFactory outer_factory;
      InnerFactory inner_factory;
      agency::cuda::gpu_id gpu;

      template<class Future>
      uber_future<
        typename std::result_of<Factory(Shape)>::type
      >
        operator()(Future& fut)
      {
        return fut.bulk_then(f, result_factory, shape, index_function, outer_factory, inner_factory);
      }
    };

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __AGENCY_ANNOTATION
    uber_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::gpu_id gpu)
    {
      auto visitor = bulk_then_visitor<Function,Factory,Shape,IndexFunction,OuterFactory,InnerFactory>{f,result_factory,shape,index_function,outer_factory,inner_factory,gpu};
      return agency::detail::visit(visitor, variant_);
    }


    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    struct bulk_then_and_leave_valid_visitor
    {
      Function f;
      Factory result_factory;
      Shape shape;
      IndexFunction index_function;
      OuterFactory outer_factory;
      InnerFactory inner_factory;

      template<class Future>
      uber_future<
        typename std::result_of<Factory(Shape)>::type
      >
        operator()(Future& fut)
      {
        return fut.bulk_then_and_leave_valid(f, result_factory, shape, index_function, outer_factory, inner_factory);
      }
    };

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __AGENCY_ANNOTATION
    uber_future<typename std::result_of<Factory(Shape)>::type>
      bulk_then_and_leave_valid(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::gpu_id gpu)
    {
      auto visitor = bulk_then_and_leave_valid_visitor<Function,Factory,Shape,IndexFunction,OuterFactory,InnerFactory>{f,result_factory,shape,index_function,outer_factory,inner_factory,gpu};
      return agency::detail::visit(visitor, variant_);
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
      return agency::detail::visit(visitor, variant_);
    }

    friend class shared_uber_future<T>;
};


template<class T>
class shared_uber_future
{
  private:
    std::shared_ptr<uber_future<T>> underlying_future_;

  public:
    shared_uber_future() = default;

    shared_uber_future(const shared_uber_future&) = default;

    shared_uber_future(uber_future<T>&& other)
      : underlying_future_(std::make_shared<uber_future<T>>(std::move(other)))
    {}

    shared_uber_future(shared_uber_future&& other) = default;

    ~shared_uber_future() = default;

    shared_uber_future& operator=(const shared_uber_future& other) = default;

    shared_uber_future& operator=(shared_uber_future&& other) = default;

    bool valid() const
    {
      return underlying_future_ && underlying_future_->valid();
    }

    bool is_ready() const
    {
      return underlying_future_ && underlying_future_->is_ready();
    }

    void wait() const
    {
      underlying_future_->wait();
    }

    auto get() ->
      decltype(underlying_future_->get_ref())
    {
      return underlying_future_->get_ref();
    } // end get()

    template<class... Args,
             class = typename std::enable_if<
               agency::cuda::detail::is_constructible_or_void<T,Args...>::value
             >::type>
    static shared_uber_future make_ready(Args&&... args)
    {
      return uber_future<T>::make_ready(std::forward<Args>(args)...).share();
    }

    template<class Function>
    uber_future<
      agency::detail::result_of_continuation_t<
        typename std::decay<Function>::type,
        shared_uber_future
      >
    >
      then(Function f)
    {
      // XXX what if there are no shared_uber_futures by the time the continuation runs?
      //     who owns the data?
      //     it seems like we need to introduce a copy of this shared_uber_future into
      //     a continuation dependent on the next_event

      return underlying_future_->then_and_leave_valid(f);
    }
};

template<class T>
shared_uber_future<T> uber_future<T>::share()
{
  return shared_uber_future<T>(std::move(*this));
}

