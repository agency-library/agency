#pragma once

#include <tuple>
#include <utility>
#include <type_traits>
#include <agency/coordinate.hpp>
#include <agency/coordinate/detail/shape/make_shape.hpp>
#include <agency/coordinate/detail/shape/shape_element.hpp>
#include <agency/detail/tuple/tuple_utility.hpp>
#include <agency/detail/point_size.hpp>
#include <agency/tuple.hpp>

namespace agency
{
namespace detail
{
namespace shape_cast_detail
{


template<class Shape>
struct make_shape_functor
{
  template<class... Args>
  __AGENCY_ANNOTATION
  Shape operator()(Args&&... args) const
  {
    return detail::make_shape<Shape>(std::forward<Args>(args)...);
  }
};


} // end shape_cast_detail


// reduces the dimensionality of x by eliding the last dimension
// and multiplying the second-to-last dimension by the last 
// this function always returns a tuple, even if it's a one-element tuple
template<class Point>
__AGENCY_ANNOTATION
rebind_point_size_t<
  Point,
  point_size<Point>::value - 1
> project_shape_impl(const Point& x)
{
  using result_type = rebind_point_size_t<Point,std::tuple_size<Point>::value-1>;

  auto last = detail::tuple_last(x);

  // XXX WAR nvcc 7's issue with tuple_drop_invoke
  //auto result = __tu::tuple_drop_invoke<1>(x, shape_cast_detail::make_shape_functor<result_type>());
  result_type result = __tu::tuple_take_invoke<std::tuple_size<Point>::value - 1>(x, shape_cast_detail::make_shape_functor<result_type>());

  detail::tuple_last(result) *= last;

  return result;
}


// reduces the dimensionality of shape by eliding the last dimension
// and multiplying the second-to-last dimension by the last
// this function unwraps single element tuples
template<class ShapeTuple>
__AGENCY_ANNOTATION
auto project_shape(const ShapeTuple& shape)
  -> typename std::decay<
       decltype(
         detail::unwrap_single_element_tuple(
           detail::project_shape_impl(shape)
         )
       )
     >::type
{
  return detail::unwrap_single_element_tuple(detail::project_shape_impl(shape));
}


// increases the dimensionality of x
// by appending a dimension (and setting it to 1)
template<class Point>
__AGENCY_ANNOTATION
rebind_point_size_t<
  Point,
  point_size<Point>::value + 1
> lift_shape(const Point& x)
{
  // x could be a scalar, so create an intermediate which is at least a 1-element tuple
  using intermediate_type = rebind_point_size_t<Point,point_size<Point>::value>;
  intermediate_type intermediate{x};

  using result_type = rebind_point_size_t<Point,point_size<Point>::value + 1>;

  result_type result = __tu::tuple_append_invoke(intermediate, 1, shape_cast_detail::make_shape_functor<result_type>());

  return result;
}


// shape_cast is recursive and has various overloads
// declare them here before their definitions

// Scalar -> Scalar (base case)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value == 1),
  ToShape
>::type
  shape_cast(const FromShape& x);


// case for casting two shapes of equal size (recursive case)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value > 1),
  ToShape
>::type
  shape_cast(const FromShape& x);


// downcast (recursive)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value < point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x);


// upcast (recursive)
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value > point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x);


// definitions of shape_cast follow


// terminal case for casting shapes of size 1
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value == 1),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  // we have to cast the 0th element of x to the type of ToShape's 0th element
  using to_element_type = shape_element_t<0,ToShape>;

  // x may or may not be a tuple, so it may not be safe to get() its 0th element
  // use get_if() to return x directly when x is not a tuple-like object
  to_element_type casted_element = static_cast<to_element_type>(detail::get_if<0>(x,x));

  return detail::make_shape<ToShape>(casted_element);
}

struct shape_cast_functor
{
  template<class ToShape, class FromShape>
  __AGENCY_ANNOTATION
  auto operator()(const ToShape&, const FromShape& x)
    -> decltype(
         shape_cast<ToShape>(x)
       )
  {
    return shape_cast<ToShape>(x);
  }
};


// recursive case for casting to a shape of equal size
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value == point_size<FromShape>::value) &&
  (point_size<ToShape>::value > 1),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return __tu::tuple_map_with_make(shape_cast_functor{}, shape_cast_detail::make_shape_functor<ToShape>{}, ToShape{}, x);
}


// recursive case for casting to a lower dimensional shape
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value < point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return shape_cast<ToShape>(project_shape(x));
}


// recursive case for casting to a higher dimensional shape
template<class ToShape, class FromShape>
__AGENCY_ANNOTATION
typename std::enable_if<
  (point_size<ToShape>::value > point_size<FromShape>::value),
  ToShape
>::type
  shape_cast(const FromShape& x)
{
  return shape_cast<ToShape>(lift_shape(x));
}


} // end detail
} // end agency

