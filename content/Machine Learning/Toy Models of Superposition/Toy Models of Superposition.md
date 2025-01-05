---
title: Toy Models of Superposition
---

https://transformer-circuits.pub/2022/toy_model/index.html \
https://github.com/Rhqo/Toy-Models-of-Superposition

> [!Abstract] Abstract
> Neural networks often pack many unrelated concepts into a single neuron – a puzzling phenomenon known as 'polysemanticity' which makes interpretability much more challenging. This paper provides a toy model where polysemanticity can be fully understood, arising as a result of models storing additional sparse features in "superposition." We demonstrate the existence of a phase change, a surprising connection to the geometry of uniform polytopes, and evidence of a link to adversarial examples. We also discuss potential implications for mechanistic interpretability.
> 
> ---
> 신경망은 종종 많은 관련 없는 개념들을 하나의 뉴런으로 포장하는데, 이는 해석 가능성을 훨씬 더 어렵게 만드는 수수께끼 같은 현상인 'polysemanticity'로 알려져 있습니다. 이 논문은 모델들이 추가적인 sparse한 특징들을 "superposition"에 저장함으로써 발생하는 polysemanticity을 완전히 이해할 수 있는 장난감 모델을 제공합니다. 우리는 상변화의 존재, 균일한 다면체의 기하학적 구조와의 놀라운 연결, 그리고 적대적 예제와의 연관성을 입증합니다. 또한 기계론적 해석 가능성에 대한 잠재적인 함의에 대해서도 논의합니다.


# Intro

> **Why is it that neurons sometimes align with features and sometimes don't?** \
> **Why do some models and tasks have many of these clean neurons, while they're vanishingly rare in others?** \
> In this paper, we use toy models — small ReLU networks trained on synthetic data with sparse input features — to investigate **how and when models represent more features than they have dimensions**. We call this phenomenon superposition.
> 
> **왜 뉴런이 때때로 특징과 일치하고 때로는 그렇지 않은가?** \
> **왜 일부 모델과 작업에는 이러한 명확한 뉴런이 많이 존재하지만, 다른 모델에서는 그 수가 극히 적은가?** \
> “**==언제, 어떻게 모델이 차원에 비해 더 많은 feature를 표현할 수 있는가==**”에 대한 고찰


다섯 개의 다양한 중요성을 가진 특징들을 2차원에서 임베딩으로 훈련하고, 이후 필터링을 위해 ReLU를 추가하고, feature의 sparsity를 변화시키는 장난감 모델을 생각해 보자.

Sparsity가 증가함에 따라 feature가 어떻게 표현되는지 아래 그림으로 볼 수 있다. \
![[Toy Models of Superposition_0.png]]

**0% Sparsity**
	- 중요한 2가지 특징이 orthogonal dimension에 할당
	- 덜 중요한 특징 3가지는 0으로 매핑되어 표현되지 않음 (비활성화)
	- Sparsity가 없으므로 독립적인 특징 표현이 가능하지만, 전체 특징 공간을 비효율적으로 사용한다.
- **80% Sparsity**
	- 중요한 4가지 특징이 antipodal pairs로 표현
	- 덜 중요한 특징은 여전히 0으로 매핑 (비활성화)
	- 일부 특징은 독립적이지 못하지만, sparsity가 증가하면서 공간 사용이 더 효율적이다.
- **90% Sparsity**
	- 5개의 모든 특징이 오각형으로 표현
	- 특징 간에 positive interference가 발생한다. (한 표현의 특징이 다른 표현의 특징에 영향을 미친다.)