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

---

Vision 모델에서는 이미지의 대부분의 위치에 수평 모서리, 곡선, 또는 개의 머리가 포함되지 않으며, 언어에서는 대부분의 토큰이 마틴 루터 킹을 언급하지 않거나 음악을 설명하는 절의 일부가 아니다. 이 아이디어는 시각과 자연 이미지 통계에 대한 고전적인 연구로 거슬러 올라간다.

![[What is Sparsity?_0.png]]

![[What is Sparsity?_1.png]]

> **왜 희소성(sparseness)인가?**
> 
> 희소성이 인공지능(AI)에 적합한 사전 분포(prior)라고 추측하는 이유는, **자연 이미지가 일반적으로 소수의 구조적 기본 요소들**(예: 모서리, 선분, 혹은 다른 기본적인 특징들)**로 묘사될 수 있다는 직관에 기반**한다 (Field, 1994). 또한, 로그-가보(Log-Gabor) 필터 세트를 이용해 이미지를 필터링하고, 그 결과 출력 분포의 히스토그램을 수집하면, 이 분포들이 주로 **높은 kurtosis를 보인다는 사실에서 희소적 구조의 증거를 확인**할 수 있다 (Field, 1993). 높은 kurtosis는 희소적 구조를 나타낸다. 
> 
> 자연 이미지에 대해 희소 코딩(sparse coding)이 적합하다고 믿게 하는 또 다른 형태의 사고 과정은 코드 요소가 다중 모달(multi-modally) 분포된 대안을 상정할 때의 결과를 고려하는 것이다. 이 경우, 특정 이벤트나 이미지 특징이 두 가지 이상의 값을 자주 취하며, 중간값을 거의 갖지 않게 된다. 그러나 자연 이미지에서 이러한 예를 상상하기는 어렵다. 자연 이미지에서 더 일반적으로 나타나는 경우는 이벤트가 거의 발생하지 않다가(대부분의 시간 동안 값이 0에 가까움), 발생하더라도 연속적인 값을 따라 나타나는 경우이다. 이는 그림 1(b)에 묘사된 분포를 가져온다. 
> 
> 희소성을 추구하는 이유는 다른 곳에서 논의된 것들과는 별개의 것임에 유의하자. 예를 들어, 연상 기억(associative memory) 용량을 증가시키거나(Baum, Moody, & Wilczek, 1988), 배선 길이를 최소화하며 연관성을 형성하는 데 용이하게 만들거나(Foldiak, 1995), 혹은 대사 효율성을 높이는 것(Baddeley, 1996)과 같은 이점들이다. 이러한 이점들은 희소 코드의 명백한 장점이지만, 여기서 논의되는 기준과는 독립적이다. 만약 데이터가 실제로 다중 모달 분포에서 비롯되어 비영(zero가 아닌 값) 주위에 무거운 피크를 가지고 있다면, 희소 코드를 찾으려는 시도는 부적절한 전략이 될 것이다. 다른 말로 하면, 희소 코딩은 데이터 내 통계적으로 독립된 요소를 찾기 위한 일반적인 원칙이 아니며, 데이터가 실제로 희소적 구조를 가질 때에만 적용된다.