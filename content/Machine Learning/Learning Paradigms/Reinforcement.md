![[Reinforcement.png]]
# Why is RL different from normal supervised learning?

1. **Stochasticity**: Rewards and state transitions may be random
2. **Credit assignment**: Reward $r_t$ may not directly depend on action $a_t$
3. **Nondifferentiable**: Can’t backprop through world; can’t compute $dr_t/da_t$
4. **Nonstationary**: What the agent experiences depends on how it acts

# Markov Decision Process (MDP)

Mathematical formalization of the RL problem: A tuple (𝑆, 𝐴, 𝑅, 𝑃, 𝛾)

