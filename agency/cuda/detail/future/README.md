This directory contains the implementations of the various types of CUDA futures.

 * `deferred_future` - This type of future becomes ready when `.wait()` or `.get()` is called. The continuation is invoked in the calling thread.
 * `async_future` - This type of future tracks CUDA kernels launched on a CUDA device. The continuation becomes ready when the kernel launch completes.
 * `future` - This type of future is conceptually a `variant<async_future, deferred_future>`.
 * `shared_future` - This type of future is conceptually a `shared_ptr<future>`.

