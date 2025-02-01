![[Reinforcement.png]]
# Why is RL different from normal supervised learning?

1. **Stochasticity**: Rewards and state transitions may be random \
	Reward와 state transition은 무작위적일 수 있다. \
	(동일한 action을 취하더라도 reward와 state transition이 확정적이지 않고 확률적으로 결정됨)
	
2. **Credit assignment**: Reward $r_t$ may not directly depend on action $a_t$ \
	특정 시점의 보상 $r_t$는 그 시점의 행동 $a_t$에 직접적으로 의존하지 않을 수 있다. \
	(Agent가 한 action에 따라 결국 얻은 reward가 언제, 어떻게 유발되었는지를 분석하는 일이 어려움)
	
3. **Nondifferentiable**: Can’t backprop through world; can’t compute $dr_t/da_t$ \
	세상의 동작 방식(world dynamics)은 미분가능하지 않을 수 있다. \
	(Reward $r_t$를 action $a_t$에 대해 미분 $dr_t/da_t$할 수 없으므로, 신경망 학습에서 흔히 사용하는 backpropagation 방법을 직접적으로 적용할 수 없음)
	
4. **Nonstationary**: What the agent experiences depends on how it acts
	Agent가 경험하는 환경은 agent가 어떤 action을 취하느냐에 따라 변할 수 있다. \
	 (Agent의 action은 환경의 state 분포와 reward 시스템에 영향을 줄 수 있으므로, 환경이 고정되어 있지 않은 dynamic system임)

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
목표는 **cumulative discounted reward를 최대화하는 optimal 정책 $\pi^*$를 찾는 것**이다.
$$
	\sum_t\gamma^tr_t
$$

- Time step t = 0일때, environment에서 initial state를 sampling: $s_0 \sim p(s_0)$
- for t = 0 until done:
	- Agent가 action을 선택: $a_t \sim \pi(a|s_t)$
	- Environment가 reward를 sampling: $r_t \sim R(r|s_t,a_t)$
	- Environment가 next state를 sampling: $s_t \sim P(s|s_t,a_t)$
	- Agent가 reward $r_t$ 와 next state $s_{t+1}$ 을 받음

# Finding Optimal Policies

