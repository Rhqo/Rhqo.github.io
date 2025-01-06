논문에서 계속해서 언급하는 "sparsity"에 대해서 다루고 넘어가고자 한다. \
Sparsity $S$를 다음과 같이 정의한다. \

> Concretely, our synthetic data is defined as follows: The input vectors $x$ are synthetic data intended to simulate the properties we believe the true underlying features of our task have. We consider each dimension $x_i$ to be a "feature". Each one has an associated sparsity $S_i$ and importance $I_i$. We let $x_i = 0$ with probability $S_i$, but is otherwise uniformly distributed between $[0,1]$. In practice, we focus on the case where all features have the same sparsity, $S = Si$
> 
> ---
> 구체적으로, 우리의 합성 데이터는 다음과 같이 정의됩니다: 입력 벡터 $x$는 우리가 작업의 진정한 기본 특징이 가지고 있다고 믿는 속성을 시뮬레이션하기 위한 합성 데이터입니다. 우리는 각 차원 $x_i$를 "특징"으로 간주합니다. 각 차원은 관련된 희소성 $S_i$와 중요도 $I_i$를 가집니다. 우리는 확률 $S_i$로 $x_i = 0$을 두었지만, 그렇지 않으면 $[0,1]$ 사이에 균일하게 분포되어 있습니다. 실제로, 우리는 모든 특징이 동일한 희소성을 가지는 경우 $S = Si$에 초점을 맞춥니다


Toy Model들이 학습하는 synthetic data는 특정 sparsity를 가지게 된다.\
여기서 sparsity는 각 feature가 0이 될 확률을 의미한다.

![[s0.gif]]

![[s0.5.gif]]

![[s0.9.gif]]

