![[Reinforcement.png]]
# Why is RL different from normal supervised learning?

1. **Stochasticity**: Rewards and state transitions may be random
2. **Credit assignment**: Reward $r_t$ may not directly depend on action $a_t$
3. **Nondifferentiable**: Can’t backprop through world; can’t compute $dr_t/da_t$
4. **Nonstationary**: What the agent experiences depends on how it acts

# Markov Decision Process (MDP)

Mathematical formalization of the RL problem: A tuple (𝑆, 𝐴, 𝑅, 𝑃, $\gamma$ )
- S : Set of possible states
- A : Set of possible actions
- R : Distribution of reward given (state, action) pair
- P : Transition probability: distribution over next state given (state, action)
- $\gamma$ : Discount factor (tradeoff between future and present rewards)

Markov Property: The current state completely characterizes the state of the world. \
Rewards and next states depend only on current state, not history. \
현재 state는 world의 state를 완전히 특징짓는다. \
Reward와 다음 state는 과거가 아닌 현재 state에만 의존한다.

Agent는 state에 따라 action의 distribution를 제공하는 Policy $\pi$ 를 수행한다. \
목표는 cumulative discounted reward를 최대화하는 optimal 정책 $\pi^*$를 찾는 것이다.
$$
	\sum_t\gamma^tr_t
$$
