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
  }
};


auto grid(size_t num_blocks, size_t num_threads)
  -> decltype(
       agency::cuda::par(num_blocks, agency::cuda::con(num_threads))
     )
{
  return agency::cuda::par(num_blocks, agency::cuda::con(num_threads));
}


//using parallel_execution_policy_2d = agency::cuda::detail::basic_execution_policy<agency::cuda::parallel_agent_2d, agency::cuda::parallel_executor>;
//const parallel_execution_policy_2d par_2d{};
//
//using concurrent_execution_policy_2d = agency::cuda::detail::basic_execution_policy<agency::cuda::concurrent_agent_2d, agency::cuda::block_executor>;
//const concurrent_execution_policy_2d con_2d{};
//
//// XXX eliminate the need for these specializations
//namespace agency
//{
//
//template<>
//struct is_execution_policy<parallel_execution_policy_2d> : std::true_type {};
//
//template<>
//struct is_execution_policy<concurrent_execution_policy_2d> : std::true_type {};
//
//}
//
//
//auto grid(agency::size2 num_blocks, agency::size2 num_threads)
//  -> decltype(
//       par_2d(num_blocks, con_2d(num_threads))
//     )
//{
//  return par_2d(num_blocks, con_2d(num_threads));
//}


int main()
{
  bulk_invoke(grid(2,32), kernel());

  // XXX this takes too long to compile
  //agency::cuda::bulk_invoke(grid({1,2}, {1,32}), kernel());

  return 0;
}

