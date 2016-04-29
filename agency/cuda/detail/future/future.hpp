#pragma once

#include <agency/detail/config.hpp>
#include <agency/cuda/detail/future/async_future.hpp>
#include <agency/cuda/detail/future/deferred_future.hpp>
#include <agency/detail/variant.hpp>
#include <agency/detail/type_traits.hpp>
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

    using variant_type = agency::detail::variant<agency::cuda::async_future<T>, agency::cuda::deferred_future<T>>;

  public:
    __AGENCY_ANNOTATION
    future() = default;

    __AGENCY_ANNOTATION
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

      __AGENCY_ANNOTATION
      future operator()(deferred_future<U>& other) const
      {
        return deferred_future<T>(std::move(other));
      }
    };

    template<class U>
    __AGENCY_ANNOTATION
    static future converting_move_construct(future<U>&& other)
    {
      return agency::detail::visit(converting_move_construct_visitor<U>{}, other.variant_);
    }

  public:
    // this constructor allows move construction of our T from a U
    // when that is possible for all the alternative futures
    template<class U,
             class = typename std::enable_if<
               std::is_constructible<
                 async_future<T>,async_future<U>&&
               >::value
               && std::is_constructible<
                 deferred_future<T>,deferred_future<U>&&
               >::value
             >::type>
    __AGENCY_ANNOTATION
    future(future<U>&& other)
      : future(converting_move_construct(std::move(other)))
    {}

    __AGENCY_ANNOTATION
    future& operator=(future&& other) = default;

    shared_future<T> share();

    template<class Future>
    __AGENCY_ANNOTATION
    Future& get()
    {
      return agency::detail::get<Future>(variant_);
    }

    template<class Future>
    __AGENCY_ANNOTATION
    const Future& get() const
    {
      return agency::detail::get<Future>(variant_);
    }

    template<class Future>
    __AGENCY_ANNOTATION
    Future&& get() &&
    {
      return agency::detail::get<Future>(std::move(variant_));
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
        return fut.then(f);
      }

      // XXX all this functor should really do is return fut_.fmap(f_)
      //     but cuda::async_future::fmap() doesn't exist at the time of writing
      struct get_and_invoke_functor
      {
        agency::cuda::async_future<T> fut_;
        Function f_;

        using result_type = agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::async_future<T>
        >;

        __AGENCY_ANNOTATION
        result_type impl(agency::cuda::async_future<void>&)
        {
          return agency::invoke(f_); 
        }

        template<class U>
        __AGENCY_ANNOTATION
        result_type impl(agency::cuda::async_future<U>& fut)
        {
          U arg = fut.get();
          return agency::invoke(f_, arg);
        }

        __AGENCY_ANNOTATION
        result_type operator()()
        {
          return impl(fut_);
        }
      };

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
                            std::is_move_constructible<Function1>::value
                          >::type* = 0)
      {
        // we can't move f into a CUDA kernel parameter at the time of writing
        // implement the following workaround:
        // "cast" fut to a deferred_future by calling .then()
        // on a ready deferred_future which waits on fut
        auto ready = agency::cuda::deferred_future<void>::make_ready();

        get_and_invoke_functor continuation{std::move(fut), std::move(f)};
        return ready.then(std::move(continuation));

        // XXX the implementation above is not efficient because these
        //     dependencies are hidden from the CUDA runtime
        //     we should investigate a way to hack CUDA stream callbacks to implement
        //     this form of .then() in cuda::async_future<T>
        //return fut.then(std::move(f));
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
      return agency::detail::visit(visitor, variant_);
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
        async_future_impl(agency::cuda::async_future<T>& fut, Function1& f,
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
        detail::throw_runtime_error("future::then_and_leave_valid_visitor::operator()(cuda::future): unimplemented");

        using result_type = agency::detail::result_of_continuation_t<
          Function,
          agency::cuda::async_future<T>
        >;

        return future<result_type>();

        // XXX TODO: implement a similar workaround as used in the then_future via casting to deferred_future
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
      agency::cuda::device_id device;

      template<class Future>
      future<
        agency::detail::result_of_t<Factory(Shape)>
      >
        operator()(Future& fut)
      {
        return fut.bulk_then(f, result_factory, shape, index_function, outer_factory, inner_factory);
      }
    };

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __AGENCY_ANNOTATION
    future<agency::detail::result_of_t<Factory(Shape)>>
      bulk_then(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::device_id device)
    {
      auto visitor = bulk_then_visitor<Function,Factory,Shape,IndexFunction,OuterFactory,InnerFactory>{f,result_factory,shape,index_function,outer_factory,inner_factory,device};
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
      agency::cuda::device_id device;

      template<class Future>
      __AGENCY_ANNOTATION
      future<
        agency::detail::result_of_t<Factory(Shape)>
      >
        operator()(Future& fut)
      {
        return fut.bulk_then_and_leave_valid(f, result_factory, shape, index_function, outer_factory, inner_factory, device);
      }
    };

    template<class Function, class Factory, class Shape, class IndexFunction, class OuterFactory, class InnerFactory>
    __AGENCY_ANNOTATION
    future<agency::detail::result_of_t<Factory(Shape)>>
      bulk_then_and_leave_valid(Function f, Factory result_factory, Shape shape, IndexFunction index_function, OuterFactory outer_factory, InnerFactory inner_factory, agency::cuda::device_id device)
    {
      auto visitor = bulk_then_and_leave_valid_visitor<Function,Factory,Shape,IndexFunction,OuterFactory,InnerFactory>{f,result_factory,shape,index_function,outer_factory,inner_factory,device};
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

    friend class shared_future<T>;
};


} // end cuda
} // end agency

