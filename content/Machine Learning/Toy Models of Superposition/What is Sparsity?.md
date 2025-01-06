논문에서 계속해서 언급하는 "sparsity"에 대해서 다루고 넘어가고자 한다. \
Sparsity $S$를 다음과 같이 정의한다. 

> Concretely, our synthetic data is defined as follows: The input vectors $x$ are synthetic data intended to simulate the properties we believe the true underlying features of our task have. We consider each dimension $x_i$ to be a "feature". Each one has an associated sparsity $S_i$ and importance $I_i$. We let $x_i = 0$ with probability $S_i$, but is otherwise uniformly distributed between $[0,1]$. In practice, we focus on the case where all features have the same sparsity, $S = Si$
> 
> ---
> 구체적으로, 우리의 합성 데이터는 다음과 같이 정의된다: 입력 벡터 $x$는 우리가 작업의 진정한 기본 feature가 가지고 있다고 믿는 속성을 시뮬레이션하기 위한 합성 데이터이다. 우리는 각 차원 $x_i$를 "feature"로 간주한다. 각 차원은 관련된 희소성 $S_i$와 중요도 $I_i$를 가진다. 확률 $S_i$에 따라 $x_i = 0$을 두었고, 그렇지 않으면 $[0,1]$ 사이에 균일하게 분포되어 있다. 실제로, 우리는 모든 특징이 동일한 희소성을 가지는 경우 $S = Si$에 초점을 맞춘다.


Toy Model들이 학습하는 synthetic data는 특정 sparsity를 가지게 된다.\
여기서 sparsity는 각 feature가 0이 될 확률을 의미한다. \
만약 feature가 3개인 어떤 synthetic data를 시각화해보면, sparsity에 따른 데이터 분포는 다음과 같을 것이다.

$S=0$
![[s0.gif]]
$S=0.5$
![[s0.5.gif]]
$S=0.9$
![[s0.9.gif]]

그렇다면, synthetic data를 이렇게 설정한 이유는 무엇일까. \
Sparsity는 원본 데이터에서 내가 찾고자 하는 특성이 얼마나 나타나는지에 대한 지표가 될 것이다. \
예를 들면, 강아지를 찾는 모델에서, 모델이 강아지 사진을 보며 강아지라고 판단짓는 근거는 원본데이터의 극히 일부가 될 것이다. \
$255\times255\times3$의 원본 데이터를 $512$차원, 혹은 그 이하의 차원을 줄여도 원본 데이터의 정보를 담고 있을 수 있는 이유가 무엇일까 생각해보면, 원본 데이터가 극히 sparse하기 때문임을 알 수 있다. \
요약하자면, sparsity는 원본 데이터에서 내가 찾고자 하는 정보(유용한 정보)의 비율이 될 것이다. 

후속 연구에서는, 원본 데이터를 차원축소한 결과인 단어 임베딩들을 sparse autoencoder를 사용하여 해석하고자 하기도 한다.