#include <agency/agency.hpp>
#include <atomic>

template<class ExecutionPolicy>
void test(ExecutionPolicy policy)
{
  using agent = typename ExecutionPolicy::execution_agent_type;
  using agent_traits = agency::execution_agent_traits<agent>;

  {
    // bulk_invoke with no parameters

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy, [&](agent&)
    {
      ++counter;
    });

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents);
  }

  {
    // bulk_invoke with one parameter

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy,
      [&](agent&, int val)
      {
        counter += val;
      },
      val
    );

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }

  {
    // bulk_invoke with one shared parameter

    int val = 13;

    std::atomic<int> counter{0};

    auto f = agency::bulk_async(policy,
      [&](agent&, int& val)
      {
        counter += val;
      },
      agency::share(val)
    );

    f.wait();

    int num_agents = agent_traits::domain(policy.param()).size();

    assert(counter == num_agents * 13);
  }
}

int main()
{
  using namespace agency;

  test(seq(10));
  test(con(10));
  test(par(10));

  test(seq(10, seq(10)));
  test(seq(10, par(10)));
  test(seq(10, con(10)));

  test(con(10, seq(10)));
  test(con(10, par(10)));
  test(con(10, con(10)));

  test(par(10, seq(10)));
  test(par(10, con(10)));
  test(par(10, par(10)));

  std::cout << "OK" << std::endl;

  return 0;
}

