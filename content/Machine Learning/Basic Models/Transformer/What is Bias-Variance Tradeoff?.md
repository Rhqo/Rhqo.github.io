---
title: What is Bias-Variance Tradeoff?
---
좋은 모델이란 무엇을 의미할까?

다음과 같은 2가지 조건이 좋은 모델을 의미한다.

1. 모델은 training dataset에 대해서 잘 설명할 수 있어야 한다.
2. 모델은 test dataset에 대해서 잘 설명할 수 있어야 한다.

Training dataset을 잘 설명하기 위해서는, 모델이 high-dimension일수록 유리하다. \
하지만, **dimension이 너무 높아지면 overfitting의 문제**가 발생한다.

Test dataset을 잘 설명하기 위해서는, 모델이 low-dimension인 것이 유리하다. \
(모델의 단순화, 경향성 파악 -> 예측에 있어서는 low-dim이 유리)\
하지만, **dimension이 낮아지면 underfitting의 문제**가 발생한다.

Overfitting이 발생하면 데이터의 작은 변동에도 모델이 민감하게 반응한다. \
이는 training data가 조금만 바뀌더라도 학습 결과가 완전히 달라질 수 있다는 의미이다. \
이러한 경우를 통계학과 기계학습 분야에서는 **모델의 variance가 크다**고 한다.

반대로, underfitting은 잘못된 가정으로 모델을 지나치게 단순화함으로 인해 입출력 데이터 간의 적절한 관계를 놓치게 만든다. \
이는 Training data의 특성이 바뀌더라도 학습 결과는 거의 달라지지 않을 수 있다는 의미이다. \
이 경우를 **모델의 bias가 크다**고 한다.

## Bias-Variance Decomposition

먼저 noise가 포함된 $y=f(x)+e$ 의 관계를 만족하는 어떠한 모델을 생각해보자.

여기서 noise e는 $\mu = 0$ 이고 variance가 $\sigma^2$ 인 정규분포를 따른다. \
따라서 $E[y]=f(x)$ 이다. \
$y$ 를 추정하기 위해, 학습 알고리즘을 통해 실제 관계함수인 $y=f(x)$의 근사함수 $\hat f​(x)=\hat f$​ 를 찾았다고 가정해보자.

어떤 확률변수 $X$ 에 대해서,
$$
Var[X]=E[(X-E[X])^2]=E[X^2]-E[X]^2
$$
$f$ 결정되어 있는, deterministic 이므로,
$$
\begin{split}
Var[f]&=E[(f−E[f])^2] = 0 , \\
E[f]&= f
\end{split}
$$
를 만족하고, $y$의 variance는,

$$
\begin{equation}
\begin{split}
Var[y]​&=E[(y−E[y])^2]\\
&=E[(y−f)^2]\\
&=E[(f+e−f)^2]\\
&=E[e^2]\\
&=Var[e]+E[e]^2\\
&=\sigma^2
\end{split}
\end{equation}​
$$

공분산 Covariance의 정의: $Cov⁡(X,Y)≡E⁡[(X−E⁡[X])(Y−E⁡[Y])]$ \
$f$와 $\hat f$ 는 서로 독립이므로 $Cov[f,\hat f]=E[f\hat f]−E[f]E[\hat f] = 0$ 이 성립하여, \
MSE의 기댓값은 다음과 같이 유도될 수 있다.
$$
\begin{equation}
\begin{split}
E[(y−\hat f)^2]​&=E[y^2+\hat f^2−2y\hat f] \\
&=E[y^2]+E[\hat f^2]−2E[y\hat y​] \\
&=Var[y]+E[y]^2+Var[\hat f​]+E[\hat f​]^2−2E[(f+e)\hat f​] \\
&=Var[y]+Var[\hat f​]+f^2+E[\hat f]^2−2E[f\hat f​] \\
&=Var[y]+Var[\hat f​]+f^2+E[\hat f​]^2−2(Cov[f,\hat f]​+E[f]E[\hat f​]) \\
&=\sigma^2+Var[\hat f]+(f−E[\hat f​])^2 \\
&=\sigma^2+Var[\hat f]+bias[\hat f​]^2
\end{split}
\end{equation}​
$$
위 식으로부터 MSE는 irreducible error와 모델의 variance, 그리고 모델의 bias의 제곱에 의해 결정됨을 알 수 있다.

![[What is Bias-Variance Tradeoff?_0.png]]
왼쪽은 high bias로 인한 underfitting이 일어나는 곳이고, 오른쪽은 high variance로 인한 overfitting이 일어나는 곳이다.