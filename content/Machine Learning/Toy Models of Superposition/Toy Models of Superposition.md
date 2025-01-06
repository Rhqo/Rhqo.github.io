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

**80% Sparsity**
- 중요한 4가지 특징이 antipodal pairs로 표현
- 덜 중요한 특징은 여전히 0으로 매핑 (비활성화)
- 일부 특징은 독립적이지 못하지만, sparsity가 증가하면서 공간 사용이 더 효율적이다.

**90% Sparsity**
- 5개의 모든 특징이 오각형으로 표현
- 특징 간에 positive interference가 발생한다. (한 표현의 특징이 다른 표현의 특징에 영향을 미친다.)


Sparsity가 낮을수록 중요한 특징이 독립적으로 표현되므로 해석 가능성이 높다.

Sparsity가 높을수록 신경망은 더 많은 특징을 하나의 뉴런에 압축하여 표현할 수 있다. 이는 신경망이 더 효율적으로 정보를 저장하고 처리할 수 있게 하지만, 특징들 간의 간섭이 발생할 수 있다.

우리는 장난감 모델들을 통해 다음과 같은 결과를 얻을 수 있었다.

> [!Results] Key Results
> - **Superposition is a real, observed phenomenon**
> - **Both monosematic and polysemantic neurons can form**
> - **At least some kinds of computation can be performed in superposition**
> - **Whether features are stored in superposition is governed by a phase change**
> - **Superposition organizes features into geometric structures** such as digons, triangles, pentagons, and tetrahedrons. 
> 
> ---
> - **Superposition은 실제로 관찰되는 현상이다**
> - **Monosemantic 뉴런과 polysemantic 뉴런 모두 형성될 수 있다**
> - **적어도 몇 가지의 계산이 superposition된 상태에서 수행될 수 있다**
> - **특징이 superposition 형태로 저장되는지의 여부는 상 변화에 의해 관찰된다**
> - **Superposition은** 이각형, 삼각형, 오각형, 사면체와 같은 **기하학적 구조로 특징들을 정리한다**
> 


# Definitions and Motivation: Features, Directions and Superposition

> In our work, we often think of neural networks as having features of the input represented as **directions** in activation space. This isn't a trivial claim. It isn't obvious what kind of structure we should expect neural network representations to have. ... \
> Despite this, we believe this kind of "**linear representation hypothesis**" is supported both by significant empirical findings and theoretical arguments
> 
> 우리의 연구에서는 신경망을 활성화 공간에서 입력의 특징이 **방향**으로 표현되는 것으로 생각하는 경우가 많다. 신경망 표현이 어떤 구조를 가질 것으로 기대해야 하는지는 분명하지 않다. ... \
> 그럼에도 불구하고, 이러한 종류의 "**linear representation hypothesis**"가 중요한 경험적 발견과 이론적 논거에 의해 뒷받침된다고 믿는다.

Linear representation hypothesis는 high-level의 concept들이 어떤 표현 공간에서 방향으로 선형적으로 표현된다는 개념이다. 이는 2가지 속성을 가지고 있다.

- **Decomposability** 
	
	Network의 표현은 독립적으로 이해할 수 있는 특징으로 설명할 수 있다.
	

- **Linearity** 
	
	Feature들은 방향으로 표현된다.
	

**Decomposability**는 모든 것을 우리 머리 속에 넣지 않고도 모델을 이해할 수 있게 한다. \
하지만, 분해 가능한 것만으론 충분하지 않고, 어떻게든 분해에 접근해야 하는데, 이를 수행하려면, representation 내의 개별 feature를 식별해야 한다. \
Linear representation에서 이것은, 활성화 공간의 **어떤 방향**이 **입력의 독립적인 feature**에 해당하는지 결정하는 것에 해당한다.

때로는, feature가 뉴런들과 일치하는 것 처럼 보이기 때문에 feature의 방향을 식별하는 것은 쉽다. \
하지만, 왜 우리는 때때로 이 매우 유용한 속성을 얻지만, 다른 경우에는 그렇지 않을까? \
우리는 이에 대해 2가지의 상쇄되는 힘이 있기 때문이라는 가정을 세웠다. 

- **Privileged Basis (Feature들을 기저 방향과 정렬하도록 유도하는 힘)**
    
    일부 표현만이 privileged basis를 가지고 있으며, 이는 **feature가 basis의 방향과 일치하도록 유도한다** (즉, 뉴런에 대응하도록 유도한다)
    
- **Superposition (Feature들이 뉴런과 대응되지 않도록 밀어내는 힘)**
    
    선형 표현은 차원보다 더 많은 특징을 나타낼 수 있으며, 우리는 **superposition**이라고 부르는 전략을 사용한다. 이는 더 큰 네트워크를 시뮬레이션하는 신경망으로 볼 수 있다. 이는 **feature를 뉴런과 일치시키는 것에서 밀어낸다.**

Superposition은 이전부터 연구되어 왔지만, 신경망에서 명확하게 발생하는 것이 입증되지는 않았다. \
이 논문은 이를 입증하고, privileged basis와 어떻게 상호작용 하는지를 탐구하는 것이다. \
만약 superposition이 network에서 실제로 발생한다면, 그것은 해석 가능성 연구를 위한 접근 방식에 깊이있는 영향을 미칠 것이며, 따라서 명확한 입증이 중요할 것이다.

## Empirical Phenomena

"Feature"와 표현 방식을 논할 때는 여러 관찰된 경험적 현상에 기반한 이론을 만든다. 이를 개념화하기 전에, 우리의 접근 방식에 영향을 준 주요 결과들을 살펴보고자 한다.

- **Word Embeddings**
    
    단어 임베딩에는 의미적 속성에 대응하는 **방향**이 존재, 이를 통해 임베딩 산술 벡터 연산이 가능하다. ([Mikolov et al](https://aclanthology.org/N13-1090.pdf)) \
    ex) V("king") - V("man") + V("woman") = V("queen")
    
- **Latent Spaces**
    
    GAN에서도 비슷한 "**벡터 연산**"과 **interpretable한 방향성**이 발견되었다. ([Decoding The Thought Vector](https://gabgoh.github.io/ThoughtVectors/))
    
- **Interpretable Neurons**
    
    **많은 연구에서 해석 가능한 뉴런들**이 발견되었으며, 이들은 **이해 가능한 특성에 반응**한다. 이에 대한 회의적인 시각이 있어, 일부 연구들은 **특정 뉴런들의 상세한 분석**을 통해 **이해 가능한 특성 감지**를 입증하고자 했다.
    
- **Universality**
    
    **동일한 특성에 반응**하는 유사한 뉴런들이 **여러 네트워크**에서 발견될 수 있다. ([Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/))
    
- **Polysemantic Neurons**
    
    동시에, 입력의 interpretable한 특성에 반응하지 **않는** 것처럼 보이는 많은 뉴런들이 있으며, 특히 서로 **관련 없는 입력들의 혼합에 반응**하는 것으로 보이는 다수의 **polysemantic 뉴런들이 존재**한다.


결과적으로 위와 같은 경험적인 현상들에 의해, 우리는 신경망 표현이 방향으로 표현된 특성으로 구성되어 있다고 생각했다.

## What are Features?

우리가 결론지은 feature의 정의는, 관찰하는 input의 interpretable한 속성 (또는 단어 임베딩 방향)이다. \
하지만 'feature'에 대한 만족스러운 정의를 만드는 것은 매우 어려운 일이고, 우리가 확신하는 단일 정의를 제시하는 대신, 세 가지 잠재적인 실용적 정의를 고려해보고자 한다.

- **Features as arbitrary functions (임의의 함수로써의 feature)**
	
	Feature를 입력의 임의 함수로 정의하는 것은 불충분하다. 관찰된 feature들은 데이터에 대한 기본적 추상화이며 여러 모델에서 일관되게 나타난다. 또한 개별적으로 구분 가능하다 - 예를 들어 '고양이'와 '자동차'는 개별 feature지만, '고양이+자동차'는 feature의 조합이다.
	
- **Features as interpretable properties (해석가능한 속성으로써의 feature)**
	
	설명된 feature들은 모두 인간이 이해하기 쉽다. 이를 "인간이 이해할 수 있는 개념의 존재"로 정의할 수 있지만, AlphaFold와 같은 AI가 발견하는 단백질 구조처럼 우리가 처음에는 이해하지 못하는 특징들도 포함해야 한다.
	
- **Neurons in Sufficiently Large Models (충분히 큰 모델에서의 뉴런)**
	
	마지막으로, 충분히 큰 신경망이 특정 뉴런을 할당하여 표현하는 입력의 속성을 feature로 정의할 수 있다. Curve detectors처럼 정교한 비전 모델에서 안정적으로 나타나는 것이 그 예시이다. 현재는 polysemantic 뉴런에서만 관찰되는 속성들도 충분히 큰 모델에서는 전용 뉴런이 생길 것으로 기대된다. 이는 순환적이지만 이전 정의들의 문제점을 해결한다.
	
