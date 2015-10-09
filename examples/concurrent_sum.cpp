#include <agency/execution_policy.hpp>
#include <vector>

int sum(const std::vector<int>& data)
{
  using namespace agency;

  int result = 0;

  bulk_invoke(con(data.size() / 2), [&](concurrent_agent& self, std::vector<int>& scratch)
  {
    auto i = self.index();
    auto n = scratch.size();

    while(n > 1)
    {
      if(i < n/2)
      {
        scratch[i] += scratch[n - i - 1];
      }

      // wait for every agent in the group to reach this point
      self.wait();

      // cut the number of active agents in half
      n -= n/2;

      // the first agent stores the result
      if(i == 0)
      {
        result = scratch[0];
      }
    }
  },
  share<0>(data));

  return result;
}

int main()
{
  size_t n = 10;

  std::vector<int> data(n, 1);

  auto result = sum(data);

  std::cout << "sum is " << result << std::endl;

  assert(result == n);

  std::cout << "OK" << std::endl;

  return 0;
}

