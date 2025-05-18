---
title: Why Transformer?
---
왜 Transformer가 생겨났는지를 살펴보려면, 왜 FCNN에서 CNN으로 발전되었는지도 살펴볼 필요가 있다.

FCNN이 CNN에 비해 연산량이 많다는 단점이 있다는 것은 익히 알려진 사실이다.

하지만 FCNN은 연산량 이외에도, vision task에 적합하지 않은 몇가지 이유가 있다.

- **Overfitting** \
	FCNN은 parameter 수가 너무 많아 overfitting에 취약하다. \
	각 neuron이 이전 layer의 모든 neuron과 연결되어 parameter 수가 기하급수적으로 증가하여 모델이 training 데이터를 단순히 암기하게 되고 일반화 가능한 패턴을 학습하지 못하게 될 수 있다.
- **Spatial Understanding** \
	FCNN의 근본적인 장점이자 약점은 광범위한 연결성이다. \
	FC layer 이전 layer의 모든 뉴런과 연결되기 때문에, 특징 간의 공간적 관계를 인식하는 능력이 부족하다. \
	이는 이웃 픽셀 간의 관계가 중요한 정보를 담고 있는 이미지와 같이 고유한 공간 구조를 가진 데이터에 적합하지 않다.
- **Inefficient Parameter Usage** \
	FCNN은 특정 유형의 데이터에 내재된 패턴을 활용하지 못한다. \
	이미지와 같이 translational invariance를 가진 input의 경우, 동일한 패턴이 다른 위치에 나타날 때마다 해당 패턴을 여러 번 학습해야 하므로 parameter 수와 학습 능력이 낭비된다.
- **Dimensional Constraints** \
	FCNN은 flatten 1D 입력을 요구하므로 데이터의 spatial 또는 temporal 구조가 손실된다. 이러한 flatten 과정은 학습 과정에 유용할 수 있는 구조적 정보를 버리게 된다.

이러한 요소들을 극복하기 위해 convolution 매커니즘이 생겨났다.

CNN은 ‘locality’라는 아이디어에서 시작되어 개발되었다.

CNN은 작은 kernel을 가진 필터를 사용하여, 이미지의 서로 다른 지역을 독립적으로 안전하게 처리한다.

하지만, 이미지에는 모든 receptive field에 걸쳐 공유되어야 하는 global information이 존재한다.

CNN은 필터의 **kernel 크기를 증가**시키거나, 더 깊은 layer의 neuron receptive field를 증가시키기 위해 **layer를 쌓는 방법**으로만 정보를 globalizing하는 것이 가능하기 때문에 이러한 문제에 적합하지 않다.

다음 그림은 공간적으로 너무 멀리 떨어진 두 입력 노드($x_1$ 과 $x_7$)를 비교할 수 없는 얕은 CNN의 한계를 보여준다.

![[Why Transformer?_0.png]]
이러한 단점을 극복하고자 나온 것이 attention 매커니즘이다.

Attention은 global information을 효율적으로 처리하기 위한 전략으로, 현재 작업에 가장 중요한 신호의 부분들에만 집중하고자 한다.

이 아이디어는 인간 지각의 주의력에서 영감을 받았다. \
이미지 장면 속 자동차 색상에 관한 질문을 받으면, 우리는 그저 수동적으로 바라보는 대신 눈을 움직여 자동차를 보게 된다.

Network에서의 attention 또한 같은 직관적 아이디어를 따른다. \
$l+1$ 레이어의 뉴런 집합은 자신들의 반응을 결정하기 위해 $l$ 레이어 의 뉴런 집합에 'attend' 한다. 

만약 우리가 그 뉴런 집합에게 입력 이미지에 있는 자동차의 색상을 보고하도록 "요청"한다면, 그들은 이전 레이어에서 자동차의 색상을 표현하는 뉴런들에게 주의를 집중해야 한다.

이러한 attention 매커니즘과 mlp를 사용하여 만든 간단한 네트워크가 바로 Transformer이다.

Transformer와 attention 매커니즘은 FCNN의 global information을 얻는 광범위한 연결성이라는 이점을 포함하면서, FCNN의 단점들을 해결한 CNN의 이점을 그대로 가지고 있다.

> [!Tips] [[What is Inductive bias?]]
> Transformer는 장점만 있는 구조 같지만, 사실은 단점이 있다. \
> Transformer는 적은 Inductive bias를 가지고 있어, 충분하지 못한 양의 데이터로 학습할 때 일반화가 잘 되지 않는다.
