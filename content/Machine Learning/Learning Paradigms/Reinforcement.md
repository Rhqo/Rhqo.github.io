![[Reinforcement.png]]
# Why is RL different from normal supervised learning?

1. **Stochasticity**: Rewards and state transitions may be random
2. **Credit assignment**: Reward $r_t$ may not directly depend on action $a_t$
3. **Nondifferentiable**: Can’t backprop through world; can’t compute $dr_t/da_t$
4. **Nonstationary**: What the agent experiences depends on how it acts

# Markov Decision Process (MDP)

Mathematical formalization of the RL problem: A tuple (𝑆, 𝐴, 𝑅, 𝑃, 𝛾)
- S: Set of possible states
- A: Set of possible actions
- R: Distribution of reward given (state, action) pair
- P: Transition probability: distribution over next state given (state, action)
- 𝛾: Discount factor (tradeoff between future and present rewards)

Markov Property: The current state completely characterizes the state of the world. \
Rewards and next states depend only on current state, not history.