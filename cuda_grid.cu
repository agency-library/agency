#include <agency/cuda/execution_policy.hpp>
#include <cstdio>


struct kernel
{
  __device__
  void operator()(agency::parallel_group<agency::cuda::concurrent_agent>& self)
  {
    printf("hello from {%d, %d}\n", (int)self.outer().index(), (int)self.inner().index());
  }

  __device__
  void operator()(agency::parallel_group_2d<agency::cuda::concurrent_agent_2d>& self)
  {
    auto outer_idx = self.outer().index();
    auto inner_idx = self.inner().index();

    printf("hello from {{%d, %d}, {%d, %d}}\n", (int)outer_idx[0], (int)outer_idx[1], (int)inner_idx[0], (int)inner_idx[1]);
  }
};


auto grid(size_t num_blocks, size_t num_threads)
  -> decltype(
       agency::cuda::par(num_blocks, agency::cuda::con(num_threads))
     )
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}


namespace agency
{
namespace cuda
{


class parallel_execution_policy_2d : public detail::basic_execution_policy<cuda::parallel_agent_2d, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy_2d>
{
  public:
    using detail::basic_execution_policy<cuda::parallel_agent_2d, cuda::parallel_executor, parallel_execution_tag, parallel_execution_policy_2d>::basic_execution_policy;
};

const parallel_execution_policy_2d par_2d{};


class concurrent_execution_policy_2d : public detail::basic_execution_policy<cuda::concurrent_agent_2d, cuda::block_executor, concurrent_execution_tag, concurrent_execution_policy_2d>
{
  public:
    using detail::basic_execution_policy<cuda::concurrent_agent_2d, cuda::block_executor, concurrent_execution_tag, concurrent_execution_policy_2d>::basic_execution_policy;
};

const concurrent_execution_policy_2d con_2d{};


} // end cuda


template<>
struct is_execution_policy<cuda::parallel_execution_policy_2d> : std::true_type {};

template<>
struct is_execution_policy<cuda::concurrent_execution_policy_2d> : std::true_type {};

} // end agency


auto grid(agency::size2 num_blocks, agency::size2 num_threads)
  -> decltype(
       agency::cuda::par_2d(num_blocks, agency::cuda::con_2d(num_threads))
     )
{
  return agency::cuda::par_2d(num_blocks, agency::cuda::con_2d(num_threads));
}


int main()
{
  bulk_invoke(grid(2,32), kernel());

  agency::cuda::bulk_invoke(grid({1,2}, {1,32}), kernel());

  return 0;
}

