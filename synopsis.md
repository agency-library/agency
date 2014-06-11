// <processor>

namespace std
{

class cpu_id
{
  public:
    typedef implementation-defined native_handle_type;

    // default constructor creates a cpu_id which represents no CPU
    cpu_id();

    cpu_id(native_handle_type handle);

    // XXX std::this_thread::native_handle() is not const -- why?
    native_handle_type native_handle() const;
};


bool operator==(cpu_id lhs, const cpu_id& rhs);
bool operator!=(cpu_id lhs, cpu_id rhs);
bool operator<(cpu_id lhs, cpu_id rhs);
bool operator<=(cpu_id lhs, cpu_id rhs);
bool operator>(cpu_id lhs, cpu_id rhs);
bool operator>=(cpu_id lhs, cpu_id rhs);
ostream& operator<<(ostream &os, const cpu_id& id);


constexpr thread_local cpu_id this_cpu = implementation-defined;


template<class Function, class... Args>
std::future<typename std::result_of<Function(Args...)>::type>
  async(const cpu_id& proc, Function&& f, Args&&... args);


// XXX should this exist?
template<class Function, class... Args>
typename std::result_of<Function(Args...)>::type
  sync(const cpu_id& proc, Function&& f, Args&&... args);


class processor_id
{
  public:
    // default constructor creates a processor_id which represents no processor
    // XXX this might be a bad policy -- maybe construct a cpu_id instead
    processor_id();

    processor_id(cpu_id id);

    processor_id(implementation-defined-processor-id-type0 id);
    ...
    processor_id(implementation-defined-processor-id-typeN id);

  private:
    variant<
      implementation-defined-unknown-id-type,
      cpu_id,
      implementation-defined-processor-id-type0,
      ...
      implementation-defined-processor-id-typeN
    > index_; // exposition-only
};


bool operator==(processor_id lhs, processor_id rhs);
bool operator!=(processor_id lhs, processor_id rhs);
bool operator<(processor_id lhs, processor_id rhs);
bool operator<=(processor_id lhs, processor_id rhs);
bool operator>(processor_id lhs, processor_id rhs);
bool operator>=(processor_id lhs, processor_id rhs);
ostream& operator<<(ostream& os, const processor_id& id);

constexpr thread_local processor_id this_processor = implementation-defined;


template<class Function, class... Args>
std::future<typename std::result_of<Function(Args...)>::type>
  async(const processor_id& proc, Function&& f, Args&&... args);


// XXX should this exist?
template<class Function, class... Args>
typename std::result_of<Function(Args...)>::type
  sync(const cpu_id& proc, Function&& f, Args&&... args);


}


// <execution_group>

namespace std
{


// provides a uniform way to introspect and construct execution groups and agents
template<class ExecutionGroup>
class execution_group_traits
{
  public:
    typedef ExecutionGroup                                 group_type;
    typedef typename ExecutionGroup::param_type            param_type;
    typedef typename ExecutionGroup::child_type            child_type; // XXX rename this
    typedef typename ExecutionGroup::size_type             size_type;

    // the nesting depth of group_type
    // single agents have depth 0
    constexpr static size_t depth = see-above;
    
    // rebinds the template parameter of some execution group type,
    // e.g. turns sequential_group<this_group_type> into sequential_group<that_group_type>
    // cf. std::allocator_traits::rebind
    template<class OtherChild>
    using rebind_execution_group = see-above;

    // constructs a group object given a tuple of parameters and a tuple of indices
    // each element of each tuple is passed to the constructor of each node of the
    // grouping hierarchy 
    // XXX rename this
    // XXX make_group()?
    template<class ParamTuple, class IndexTuple>
    static group_type make(const ParamTuple& params, const IndexTuple& indices);

    // given the parameters to construct a group object, returns the range of indices
    // its children will take on
    // XXX rename this to emphasize that the range of indices is not necessarily 1D
    static unspecified-range-type range(const param_type &params); // XXX rename this
};


// represents a single sequential agent of execution
class agent
{
  public:
    typedef size_t size_type;

    size_type size() const;
    size_type index() const;

    // clients may only move construct
    agent(agent &&other);
};


// represents a single sequential group of execution agents or groups
template<class ExecutionGroup = agent>
class sequential_group
{
  public:
    typedef ExecutionGroup child_type; // XXX rename this
    typedef typename execution_group_traits<child_type>::size_type size_type;

    // the type of object encapsulating
    // allows uniform object construction
    typedef unspecified param_type;

    // clients may only move construct
    sequential_group(sequential_group &&other);

    // the range of linear indices of child groups
    // XXX instead of indices_begin() & indices_end(), we
    //     should just return a single range in the 1D case
    size_type indices_begin() const;
    size_type indices_end() const;

    size_type size() const;

    // access to the child of this sequential_group
    // XXX rename these
    child_type &child();
    const child_type &child() const;

  private:
    // only the implementation may construct sequential_group objects directly
    sequential_group(const param_type &params); // exposition only
};


// represents a single parallel group of execution agents or groups
template<class ExecutionGroup = agent> class parallel_group; // similar to sequential_group


// represents a single concurrent group of execution agents or groups
template<class ExecutionGroup = agent>
class concurrent_group
{
  public:
    typedef ExecutionGroup child_type; // XXX rename this
    typedef typename execution_group_traits<child_type>::size_type size_type;

    typedef unspecified param_type;

    // clients may only move construct
    agent(agent &&other);

    // the range of linear indices of child groups
    size_type indices_begin() const;
    size_type indices_end() const;
    size_type size() const;

    child_type &child();
    const child_type &child() const;

    // wait for all children to call this function
    // synchronize all children (barrier)
    void wait();

  private:
    // only the implementation may construct sequential_group objects directly
    concurrent_group(const param_type &params); // exposition only
    std::barrier &barrier_; // exposition only
};


// a dynamic, generic view of some execution group 
class execution_view
{
  public:
    typedef size_t size_type;

    // copies g
    execution_view(const execution_view &g);

    // moves g
    execution_view(execution_view &&g) = default;

    // creates a view of g
    template<class ExecutionGroup>
    execution_view(ExecutionGroup& g);

    execution_view &operator=(execution_view &&g) = default;

    // copy assignment
    execution_view &operator=(const execution_view &g);

    template<typename ExecutionGroup>
    execution_view& operator=(ExecutionGroup& g);

    size_type size() const;

    size_type index() const;

    // child access
    // XXX rename this
    execution_view child();
    const execution_view child() const;
};


// returns a view of the current execution group
// XXX rename this
// XXX this should probably return a value instead of a reference
execution_view& this_group();


// is_execution_policy is derived from std::true_type
// iff T is an execution policy type
// otherwise, derives from std::false_type
// only the implementation may specialize this template
template<class T> struct is_execution_policy;


template<class T>
using is_execution_policy_v = is_execution_policy<T>;

}


// <execution_policy>

namespace std
{
  
class sequential_execution_policy
{
  public:
    // XXX need to unify all of these members and specify the return types

    // nests n ExecutionPolicies in a sequential execution policy wrapper
    template<class ExecutionPolicy>
    unspecified-nested-execution-policy-type operator()(size_t n, ExecutionPolicy&& exec) const;

    // groups n ExecutionGroups in a sequential execution policy wrapper
    // XXX rename this
    template<class ExecutionGroup>
    unspecified-grouped-execution-policy-type make(size_t n) const;

    sequential_execution_policy operator()(size_t n) const;

    // XXX rename this
    template<class ExecutionGroup, class... Args>
    unspecified-grouped-execution-policy-type make(Args&&... args) const;
};


class parallel_execution_policy; // similar to sequential_execution_policy
class concurrent_execution_policy; // similar to sequential_execution_policy


// global execution policy objects 
constexpr sequential_execution_policy seq;
constexpr parallel_execution_policy   par;
constexpr concurrent_execution_policy con;


// invokes f(args...) according to exec
// communicates exceptions through exception_list
// XXX rename this
template<class ExecutionPolicy, class Function, class... Args>
void sync(ExecutionPolicy&& exec, Function&& f, Args&&... args);


// asynchronously invokes f(args...) according to exec
// communicates exceptions through exception_list
template<class ExecutionPolicy, class Function, class... Args>
future<void> async(ExecutionPolicy&& exec, Function&& f, Args&&... args);


}

