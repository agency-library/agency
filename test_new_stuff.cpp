#include <execution_agent_new>
#include <execution_policy>
#include <__nested_execution_policy_new>


namespace std
{

template<class ExecutionAgent, class BulkExecutor, class ExecutionCategory, class Function>
void bulk_invoke_new(const __basic_execution_policy<ExecutionAgent,BulkExecutor,ExecutionCategory>& exec,
                     Function f)
{
  using traits = std::execution_agent_traits_new<ExecutionAgent>;

  auto param = exec.param();
  auto shape = traits::shape(param);
  auto shared_init = traits::make_shared_initializer(param);

  using executor_index = typename executor_traits<BulkExecutor>::index_type;
  using shared_param_type = typename executor_traits<BulkExecutor>::template shared_param_type<decltype(shared_init)>;

  return executor_traits<BulkExecutor>::bulk_invoke(exec.executor(), [=](executor_index agent_idx, shared_param_type shared_params)
  {
    traits::execute(agent_idx, param, f, shared_params);
  },
  shape,
  shared_init);
}

}


int main()
{
  bulk_invoke_new(std::seq, [](std::sequential_agent_new& self)
  {
    std::cout << "self.index(): " << self.index() << std::endl;
  });

  std::mutex mut;
  bulk_invoke_new(std::con(10), [&mut](std::concurrent_agent_new& self)
  {
    mut.lock();
    std::cout << "self.index(): " << self.index() << " arriving at barrier" << std::endl;
    mut.unlock();

    self.wait();

    mut.lock();
    std::cout << "self.index(): " << self.index() << " departing barrier" << std::endl;
    mut.unlock();
  });

  using seq_seq_type = std::__nested_execution_policy_new<
    std::sequential_execution_policy,
    std::sequential_execution_policy
  >;

  auto seq_seq = seq_seq_type(std::seq(3), std::seq(2));

  bulk_invoke_new(seq_seq, [](std::sequential_group_new<std::sequential_agent_new>& self)
  {
    std::cout << "index: (" << self.index() << ", " << self.inner().index() << ")" << std::endl;
  });


  using seq_seq_seq_type = std::__nested_execution_policy_new<
    std::sequential_execution_policy,
    seq_seq_type
  >;

  auto seq_seq_seq = seq_seq_seq_type(std::seq(4), seq_seq);

  bulk_invoke_new(seq_seq_seq, [](std::sequential_group_new<std::sequential_group_new<std::sequential_agent_new>>& self)
  {
    std::cout << "index: (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ")" << std::endl;
  });

  using seq_con_type = std::__nested_execution_policy_new<
    std::sequential_execution_policy,
    std::concurrent_execution_policy
  >;

  auto seq_con = seq_con_type(std::seq(2), std::con(2));
  
  using con_seq_con_type = std::__nested_execution_policy_new<
    std::concurrent_execution_policy,
    seq_con_type
  >;

  auto con_seq_con = con_seq_con_type(std::con(2), seq_con);

  bulk_invoke_new(con_seq_con, [&mut](std::concurrent_group_new<std::sequential_group_new<std::concurrent_agent_new>>& self)
  {
    // the first agent in the first subgroup waits at the top-level barrier
    if(self.inner().index() == 0 && self.inner().inner().index() == 0)
    {
      mut.lock();
      std::cout << "(" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") arriving at top-level barrier" << std::endl;
      mut.unlock();

      self.wait();

      mut.lock();
      std::cout << "(" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") departing top-level barrier" << std::endl;
      mut.unlock();
    }

    // every agent waits at the inner most barrier
    mut.lock();
    std::cout << "  (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") arriving at bottom-level barrier" << std::endl;
    mut.unlock();

    self.inner().inner().wait();

    mut.lock();
    std::cout << "  (" << self.index() << ", " << self.inner().index() << ", " << self.inner().inner().index() << ") departing bottom-level barrier" << std::endl;
    mut.unlock();
  });

  return 0;
}

