
Deepseek-r1-zero는 Group Relative Policy Optimization (GRPO)를 사용한다. \
이는 일반적인 Reinforcement learning에서 사용하는 Critic 모델을 생략하고, \
대신 그룹의 reward들을 사용하여 optimization하는 방식

$$
	\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)]
$$
모델은 데이터셋 $P(Q)$에서 질문 $q$를 가져온다. \
이전 policy $\pi_{\theta_{old}}$에서 output을 sampling한다.

$$
	\frac{1}{G} \sum_{i=1}^G \left(
	\min \left(
	\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)} A_i,
	\text{clip} \left(
	\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},
	1 - \epsilon, 1 + \epsilon
	\right) A_i
	\right)
	- \beta \: \mathbb{D}_{\text{KL}} (\pi_\theta || \pi_{\text{ref}})
	\right)
$$

**$\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}$ 는 새로운 policy가 이전 policy에 비해 얼마나 confident한지를 물어보는 metric이다.** \
현재 policy의 확률이 높으면(1보다 크면), 이전 policy보다 좋은 답변을 출력했다는 의미일 것이고, \
반대로 1보다 작으면 이전 policy가 더 좋은 답변을 출력했다는 의미일 것이다.

여기에 Advantage $A$ 를 곱하여, reward가 그룹의 평균에 비해 높은지 여부에 따라 확률을 "**얼마나**" 증가시킬지를 결정하도록 한다.

**$\text{clip} \left(	\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)},	1 - \epsilon, 1 + \epsilon \right)$ 는 policy가 너무 크게 변경되지 않도록 막는 metric이다.** \
Policy의 ratio가 $1-\epsilon$ 보다 낮으면 $1-\epsilon$ 으로 자르고, $1+\epsilon$ 보다 높으면 $1+\epsilon$ 로 자른다. \
역시나 Advantage $A$ 를 곱한다.

**Policy의 ratio와 clip된 policy ratio의 $min$을 사용하여, ratio를 제한한다.** \
비율을 ($1-\epsilon, 1+\epsilon$) 으로 제한하여, 모델이 단일 보상에 대해 과도하게 optimize되지 않도록 한다.

**$\beta \: \mathbb{D}_{\text{KL}} (\pi_\theta || \pi_{\text{ref}})$ 는 KL Divergence 식으로, 현재 policy와 reference policy(Deepseek-v3)의 probability distribution 차이를 측정한다.** \
만약, 현재 policy가 ref policy와 너무 다르면, 값이 커지게 되고, objective 함수에서 패널티를 부여하게 될 것이다. \
이를 통해, 학습이 reference model인 Deepseek-v3와 너무 멀리 떨어지지 않게 조절하는 역할을 하게 된다. \
$\beta$ 는 하이퍼파라미터로, 기준 모델과 얼마나 가까이 유지되어야 하는지를 조절한다.

**전체의 평균 ($\frac1G\sum_{i=1}^G$)를 구하는 이유는 output이 그룹 형태이기 때문에, 특정 output에 과도하게 최적화되지 않도록 하기 위함이다.**

$$
	A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \ldots, r_G\})}{\text{std}(\{r_1, r_2, \ldots, r_G\})}
$$
Advantage $A$ 는 각 출력 $o_i$ 가 같은 그룹 내 다른 출력들과 비교했을 때 **얼마나** 좋은지 평가하는 방식이다. \
"출력 $o_i$ 의 reward인 $r_i$"에서 "그룹의 평균 reward"를 뺀 것을 "그룹의 reward의 표준편차"로 나눈다.
