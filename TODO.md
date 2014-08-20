Simplifications to agent hierarchy:
  * ~~eliminate std::agent~~
  * ~~rename groups to std::concurrent_agent, std::parallel_agent, std::sequential_agent~~
  * these agents have no .size(), but know their .group_size()

Should enable these simplifications:
  * this should greatly simplify the workarounds in execution_group_traits
  * this should also eliminate the superfluous index of the top-most group
  * solves question of what should std::agent's domain type be
  * isn't necessary to do a .child() for simple SAXPY-like apps

maybe all of these types could convert to std::agent, which would forget its children and execution category
or we could use type erasure to preserve the hierarchy somehow

Miscellaneous:
  * rename .bulk_add() to .bulk_execute()
  * eliminate __has_domain -- all agents have to have a domain

