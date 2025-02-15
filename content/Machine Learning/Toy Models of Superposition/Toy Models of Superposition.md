---
title: Toy Models of Superposition
tags:
  - subtitle
---

https://transformer-circuits.pub/2022/toy_model/index.html \
https://github.com/Rhqo/Toy-Models-of-Superposition

> [!Abstract] Abstract
> Neural networks often pack many unrelated concepts into a single neuron – a puzzling phenomenon known as 'polysemanticity' which makes interpretability much more challenging. This paper provides a toy model where polysemanticity can be fully understood, arising as a result of models storing additional sparse features in "superposition." We demonstrate the existence of a phase change, a surprising connection to the geometry of uniform polytopes, and evidence of a link to adversarial examples. We also discuss potential implications for mechanistic interpretability.
> 
> ---
> 신경망은 종종 많은 관련 없는 개념들을 하나의 뉴런으로 포장하는데, 이는 해석 가능성을 훨씬 더 어렵게 만드는 수수께끼 같은 현상인 'polysemanticity'로 알려져 있습니다. 이 논문은 모델들이 추가적인sparse한 특징들을 "superposition"에 저장함으로써 발생하는 polysemanticity을 완전히 이해할 수 있는 장난감 모델을 제공합니다. 우리는 상변화의 존재, 균일한 다면체의 기하학적 구조와의 놀라운 연결, 그리고 적대적 예제와의 연관성을 입증합니다. 또한 기계론적 해석 가능성에 대한 잠재적인 함의에 대해서도 논의합니다.


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
![[TMS_0.png]]

- **0% Sparsity**
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


Sparsity가 낮을수록 중요한 특징이 **독립적으로 표현**되므로 해석 가능성이 높다. \
Sparsity가 높을수록 신경망은 **더 많은 특징을 하나의 뉴런에 압축하여 표현**할 수 있다. 이는 신경망이 더 효율적으로 정보를 저장하고 처리할 수 있게 하지만, 특징들 간의 간섭이 발생할 수 있다.

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
	
## Features as Directions

이전 섹션에서 언급했듯이, 특징들은 방향성으로 표현된다. \
단어 임베딩에서 "성별"과 "왕족"은 방향성을 가지며 V("왕") - V("남자") + V("여자") = V("여왕")과 같은 연산이 가능하다. \
뉴런의 활성화 정도 역시 표현의 basis 방향과 대응된다.

![[TMS_1.png|width=500]]

활성화 공간에서 특징들이 방향과 대응될 때 신경망 표현을 선형이라고 하자. \
선형 표현에서는 각 특징 $f_i$이 대응되는 표현 방향 $W_i$를 갖는다. \
값 $x_{f_1}, x_{f_2}, ...$로 활성화되는 다수의 특징 $f_1, f_2, ...$의 존재는 $x_{f_1}W_{f_1} + x_{f_2}W_{f_2} ...$로 표현된다. \
분명히 하자면, 표현되는 특징들은 거의 확실히 입력의 비선형 함수이다. \
오직 feature에서 활성화 벡터로의 매핑만이 선형이다. \
어떤 것이 선형 표현인지는 무엇을 특징으로 간주하는지에 따라 달라진다는 점에 주목하자.

신경망이 실증적으로 linear representation을 가지는 것은 우연이 아니라고 생각한다. \
신경망은 **non-linearlity가 산재된 linear function들로 구성**되어 있다. \
어떤 의미에서는 선형 함수가 계산의 대부분을 차지한다(FLOPs로 측정). \
선형 표현은 신경망이 정보를 표현하는 데 있어 자연스러운 형식이다! 구체적으로 **세 가지 주요 이점**이 있다:

- **Linear representations are the natural outputs of obvious algorithms a layer might implement** \
	특정 가중치 템플릿과 패턴을 매칭하도록 뉴런을 설정하면, 자극이 템플릿과 더 잘 일치할수록 더 강하게 활성화되고 덜 일치할수록 더 약하게 활성화된다.
- **Linear representations make features “linearly accessible”** \
	일반적인 신경망 계층은 선형 함수 뒤에 비선형성이 따라오는 구조이다. 이전 계층의 특징이 선형적으로 표현되는 경우, 다음 계층의 뉴런은 이를 '선택'하여 일관되게 해당 뉴런을 활성화하거나 억제할 수 있다. 만약 특징이 비선형적으로 표현된다면, 모델은 한 단계에서 이를 수행할 수 없을 것이다.
- **Statistical Efficiency** \
	서로 다른 방향으로 특징을 표현하면 선형 변환을 가진 모델에서 non-local generalization가 가능해져 통계적 효율성이 높아진다. [Representation learning](https://arxiv.org/pdf/1206.5538)과 [Deep Learning, NLP, and Representations](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/#word-embeddings)에서 자세한 설명을 확인할 수 있다.

## Privileged vs Non-privileged Bases

Feature들이 방향으로 인코딩된다고 하더라도, 어떤 방향일 것인지에 대한 것을 생각해보고자 한다. \
어떨 때는 basis direction을 고려하는 것이 유용해 보이지만, 다른 때는 그렇지 않다. \
왜 이럴까?

![[TMS_2.png]]

연구자들이 단어 임베딩을 연구할 때 basis direction을 분석하는 것은 의미가 없다. \
**Basis dimension이 special할 것이라는 이유가 없기 때문이다**. \
예를 들면, 단어 임베딩에 임의의 linear transformation $M$을 적용하고 $M^{-1}$을 후속 가중치에 적용해보면, 이는 basis dimension이 완전히 다른 동일한 모델을 만들어낼 것이다. \
이것이 우리가 말하는 non-privileged basis이다.

하지만 많은 신경망 레이어는 이와 같지 않다. \
종종 아키텍처의 어떤 특성 때문에 **basis 방향이 special해진다**. \
예를 들어, activation function을 적용하는 것과 같은 경우이다. \
이는 "대칭을 깨뜨려" 이러한 방향을 특별하게 만들고, 잠재적으로 특징들이 basis dimensions과 정렬되도록 유도한다. \
우리는 이를 privileged basis이라고 부르며, privileged direction을 "neurons"라고 부른다. \
이러한 뉴런은 일반적으로 interpretable feature에 해당한다.

이 관점에서 볼 때, **뉴런이 interpretable한지를 묻는 것은 그것이 privileged basis에 있을 때만 의미가 있다**. \
실제로 우리는 일반적으로 "neurons"이라는 단어를 privileged basis에 있는 기준 방향에 대해서만 사용한다.

Privileged basis가 있다고 해서 특징들이 **basis-aligned된다고 보장되는 것은 아니다**. \
우리는 흔히 그렇지 않다는 것을 알게 될 것이다! \
하지만 이것은 질문이 의미가 있게 만들기 위한 최소한의 조건이 될 것이다.

| **특징**       | **Privileged Basis** | **Non-privileged Basis**                    |
| ------------ | -------------------- | ------------------------------------------- |
| **기저의 중요성**  | 특정 방향(뉴런)이 중요        | 모든 방향이 동일한 중요성                              |
| **특징 표현 방식** | 뉴런과 특징이 1:1로 매핑      | 특징이 여러 방향에 걸쳐 분산됨                           |
| **대칭성**      | 대칭성이 깨짐              | 대칭성이 유지됨                                    |
| **해석 가능성**   | 높은 해석 가능성            | 낮은 해석 가능성                                   |
| **대표적인 사례**  | CNN, MLP 뉴런          | Word Embedding, Transformer Residual Stream |

> [!Tip]
> 위의 얘기를 들어보면, non-privileged basis를 연구하는 것이 의미없어 보이지만, 사실 privileged bases 없이도 활성화 연구하는 것이 가능하다. "남자"와 "여자" 사이의 차이 벡터를 취해 단어 임베딩에서 성별 방향을 만드는 것처럼, 단지 연구할 direction을 어떻게든 찾아내기만 하면 privileged bases 없이도 활성화를 연구하는 것이 가능하다.

## The Superposition Hypothesis

Privileged basis가 있을 때조차도, 뉴런들이 "polysemantic"해서 여러 무관한 특징에 반응하는 경우가 많다. \
이에 대한 한 가지 설명은 superposition hypothesis이다. \
대략적으로 superposition의 아이디어는 신경망이 "뉴런보다 더 많은 특징을 표현하고 싶어한다"는 것이다. \
그래서 그들은 고차원 공간의 특성을 이용하여 훨씬 더 많은 뉴런을 가진 모델을 시뮬레이션한다.

![[TMS_3.png]]

몇몇의 수학적인 근거가 있다:
- **Almost Orthogonal Vectors** \
	$n$-dimensional 공간에서는 $n$개의 직교 벡터를 가질 수만 있지만, 고차원 공간에서는 $\exp(n)$ 만큼의 "almost orthogonal"한 ( $< \epsilon$ 코사인 유사도) 벡터를 가질 수 있다. \
	([Johnson-Lindenstrauss lemma](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma): 고차원 공간의 점들을 측정한 거리를 거의 보존하면서 저차원 공간에 투영할 수 있음을 보장하는 수학적 결과)
	
- **Compressed Sensing** \
	일반적으로, 벡터를 저차원 공간에 투영하면 원래 벡터를 재구성할 수 없다. 그러나 원래 벡터가 sparse하다는 것을 알면 상황이 달라지는데, 이러한 경우 종종 원래 벡터를 복구할 수 있다

구체적으로, superposition hypothesis에서는 feature가 뉴런 출력의 벡터 공간에서 almost orthogonal한 방향으로 표현된다. \
Feature가 거의 직교하기 때문에, 하나의 feature가 활성화되면 다른 feature들이 약간 활성화되는 것처럼 보인다. \
이러한 "noise” 또는 "interference"을 허용하는 데는 비용이 들지만, 신경망의 경우 굉장히 sparse한 feature를 가지고 있기 때문에, 이러한 비용은 더 많은 특징을 표현할 수 있는 이점에 의해 초과될 수 있다! \
(중요한 것은, sparsity가 크기 때문에 sparse feature는 서로 interfere를 일으키는 경우가 드물고, non-linear activation function들이 소량의 noise를 걸러내는 기회를 제공한다는 것이다.)

![[TMS_4.png]]
Superposition의 좋은 예(왼쪽)와 안좋은 예(오른쪽)

Superposition이 기대하는 대로 특징이 충분히 희소하지 않으면 간섭(interference)이 커지고, 재구성의 불확실성도 증가한다.

![[TMS_5.png]]

- **Hypothetical Disentangled Model**
    - 모든 뉴런이 각각의 독립적인 특징(disentangled feature)을 표현한다고 가정하는 이상적인 구조
    - 각 뉴런은 superposition 없이 **하나의 명확하고 독립적인 의미**를 가진다.
- **Observed Model**
    - 실제 신경망은 가설적인 네트워크의 저차원 투영(low-dimensional projection)으로 나타난다.
    - 뉴런 간에 superposition이 발생하여 **다수의 특징이 동일한 뉴런에 인코딩**된다.

## Summary: A Hierarchy of Feature Properties

- **Decomposability** \
    분해 가능한 신경망 활성화는 **다른 feature 값에 의존하지 않는 의미로서 feature로 분해될 수 있다**. \
    (이 속성은 궁극적으로 가장 중요하다 - 차원의 저주를 극복하는 데 있어 분해의 역할을 참조.)
    
- **Linearity** \
    **Feature는 방향에 해당**한다. 각 특성 $f_i$는 해당하는 표현 방향 $W_i$를 가지고 있다. 여러 특성 $f_1, f_2, ...$ 이 값 $x_{f_1}, x_{f_2}, ...$ 으로 활성화되는 경우는 $x_{f_1}W_{f_1} + x_{f_2}W_{f_2} ...$ 로 나타낸다. (**선형 결합으로 나타낼 수 있다.**)
    
- **Superposition vs Non-superposition** \
    선형 표현은 $W^TW$ 가 **가역적이지 않으면** **superposition**을 나타내고, **가역적**이라면 **superposition을 나타내지 않는다**.
    
- **Basis-Aligned** \
    표현이 basis-aligned되었다고 할 수 있는 경우, 모든 $W_i$ 가 one-hot basis 벡터이다. 모든 $W_i$ 가 sparse할 경우 표현은 부분적으로 기저 정렬되어 있다고 할 수 있다. 이는 특권 있는 기저를 필요로 한다. \
    → one-hot basis vector : 하나의 성분만 1이고 나머지 성분은 모두 0인 벡터. 특정 차원을 명확히 나타낼 수 있고, 특정 벡터 공간에서의 표준 기저 역할을 한다.
    

Decomposability, Linearity는 널리 퍼져 있다고 가정하는 속성이고, Superposition, Basis-Aligned는 때때로만 발생한다고 믿는 속성이다.

# Demonstrating Superposition

> If one takes the superposition hypothesis seriously, a natural first question is **whether neural networks can actually noisily represent more features than they have neurons**. If they can't, the superposition hypothesis may be comfortably dismissed.
> 
> Superposition hypothesis를 진지하게 받아들인다면, 자연스러운 첫 번째 질문은 **neruon들이 실제로 그들보다 더 많은 특성을 noisily하게 표현할 수 있는지 여부**이다. 만약 그렇지 않다면, 중첩 가설은 안심하고 기각될 수 있다.

Linear 모델에서의 직관은 이것이 불가능하다는 것이다: 선형 모델이 할 수 있는 최선은 주성분을 저장하는 것(PCA)이다. \
그러나 약간의 non-linearlity를 추가하면 모델이 근본적으로 다른 방식으로 작동할 수 있다! \
이것이 superposition의 첫 번째 시연이다. (또한 매우 간단한 신경망의 복잡성에 대한 교훈이 될 것이다.)

## Experiment Setup

목표는 신경망이 고차원 벡터 $x \in R^n$ 를 저차원 벡터 $h \in R^m$ 로 투영한 후 다시 복원할 수 있는지를 탐구하는 것이다.
-> $n$차원의 정보를 $m$차원에 담고자 할 때, $m$차원에 얼마나 잘 임베딩되는지를 실험

![[TMS_6.png]]

### The Feature Vector ($X$)

고차원의 벡터 $x$ : idealize된, hypothetical disentangled model의 activation.

Feature가 가상의 더 큰 모델의 뉴런과 완벽하게 align되어 있다고 상상하고 있기 때문에, 각 요소 $x_i$를 "feature"라고 부른다. \
(Vision model에서는 이것이 gabor filter, curve detector, or a floppy ear detector일 수 있고, 언어 모델에서는 특정 유명인을 언급하는 토큰이나 특정 종류의 설명이 되는 절을 나타낼 수 있다.)

현재 특성에 대한 진실의 기준이 없기 때문에, 우리는 특성이 모델링 측면에서 갖고 있다고 믿는 중요한 속성을 시뮬레이션하는 synthetic data를 생성해야 한다. \
우리는 세 가지 주요 가정을 한다:

- **Feature Sparsity** \
    자연 세계에서, **많은 feature는 드물게 발생한다는 점에서 sparse하다**. \
	따라서, synthetic data 또한 **feature에 대해 sparse distribution을 선택할 것**이다.
    
- **More Features Than Neurons** \
	모델이 표현할 수 있는 **잠재적으로 유용한 feature**가 엄청 많다. 실제 모델에서 feature와 neuron간의 이러한 불균형은 신경망 표현에서 **central tension**으로 보인다.
    
- **Features Vary in Importance** \
    **모든 features가 주어진 작업에 대해 동일하게 유용하지는 않다**. 일부는 다른 것보다 손실을 더 많이 줄일 수 있다. \
	예를 들면, 서로 다른 개 품종을 분류하는 것이 주요 작업인 ImageNet 모델의 경우, 늘어진 귀 탐지기는 그것이 가질 수 있는 가장 중요한 특성 중 하나겠지만, 다른 특성들은 성능을 아주 조금만 향상시킬 수 있다.

> [!tip] [[What is Sparsity?]]

### The Model ($X \to X'$)

우리는 아래의 두가지 모델을 고려한다. \
Linear model은 superposition이 나타나지 않는, 잘 이해되는 baseline이다. \
ReLU output model은 superposition이 나타나는 아주 간단한 모델이다. \
두 모델은 마지막 activation function만 다르다.

![[TMS_7.png]]

**Why these models?**

Superposition hypothesis에 따르면, higher-dimensional model의 각 feature는 lower-dimensional space의 direction에 해당한다. \
이것은 우리가 $h = Wx$ 의 선형 맵으로의 down projection이 가능하다는 것을 의미한다. \
각 열 $W_i$ 가 lower-dimesional space에 feature $x_i$ 를 의미한다는 것을 주목하자.

Original vector를 복원하기 위해서, 우리는 같은 행렬의 transpose $W^T$ 를 사용할 것이다. \
이것은 lower-dimensional space에서의 direction이 실제 feature에 해당하는 것인지에 관한 ambiguity를 피하는 것에 이점이 있다. \
또한 수학적으로도 상대적으로 원칙적이며, 경험적으로도 효과가 있다.

또한, bias도 추가한다. Bias는 모델이 표현하지 않는 특성을 기댓값으로 설정할 수 있도록 해 준다. 나중에 보겠지만, negative bias를 설정하는 것은 2번째 이유로 인해, superposition에 중요하다. (대략적으로 말하자면, 모델이 약간의 noise를 무시할 수 있게 해 준다.)

마지막은 activation function을 추가할 것인지의 여부이다. 이는 superposition이 발생하는지의 여부와 매우 중요하다. 실제 신경망에서 특성이 실제로 모델에 의해 계산에 사용될 때, 활성화 함수가 존재할 것이므로, 마지막에 활성화 함수를 포함하는 것이 원칙적이다.

### The Loss

Loss는 위에서 설명한 특성 중요도 $I_i$ 로 가중치가 부여된 MSE(mean squared error) 이다:
$$L = \sum_x\sum_iI_i(x_i-x'_i)^2$$

## Basic Results

첫 번째 실험은 단순하게 서로 다른 sparsity 수준을 가진 몇 가지 ReLU 출력 모델을 훈련하고 결과를 시각화하는 것이다.

가장 중요한 질문은 ‘어떻게 결과를 시각화할 것인가‘이다. \
가장 단순한 방법은 $W^TW$ (a features by features matrix)와 $b$ (a feature length vector)를 시각화하는 것이다. \
Feature들은 가장 중요한 것부터 덜 중요한 순으로 배열된다는 점에 주목하면, 결과는 꽤나 좋은 구조를 가질 것이다.

다음은 작은 모델인 ($n=20; \; m=5;)$ 인 작은 모델에 대한 예시이다. \
이 모델은 "예상되는 선형 모델과 유사한" 방식으로 동작하며, 그 차원만큼의 feature만을 나타낸다:

![[TMS_8.png]]

우리가 정말로 신경 쓰는 것은 이 hypothesize된 superposition 현상이다. \
모델이 "extra feature"을 non-orthogonal하게 저장하여 나타내는 것인가? \
좀 더 명백하게 접근할 수 있는 방법이 있을까? \
한 가지 질문은 **모델이 몇 개의 특징을 표현하는지**를 알아보는 것이다. \
특징이 표현되는지 여부는 그 임베딩 벡터의 $norm$인 $||W_i||$ 에 의해 결정된다.

우리는 또한 **주어진 특징이 다른 특징과 차원을 공유하는지**를 이해하고 싶다. \
이를 위해 우리는 $\sum_{j\not=i} (\hat{W_i}\;\cdot\;W_j)^2$ 를 계산하여 모든 다른 특징을 $W_i$ 의 방향 벡터로 사영한다. \
만약 이 값이 0이라면 해당 특징은 다른 특징들과 직교한다(아래의 짙은 파란색). \
반면에, 이 값이 1 이상이면 다른 feature의 어떤 그룹이 해당 feature만큼 강하게 활성화될 수 있다는 것을 의미한다.

![[TMS_9.png]]

다음과 같이 어떤 한 벡터가 다른 모든 벡터와 orthogonal 해야만 검정색이 될 것이다.
![[TMS_10.png|width=500]]

우리가 이전에 살펴본 모델을 이렇게 시각화할 수 있다: 이제 모델을 시각화하는 방법이 생겼으니, 실제로 실험을 시작할 수 있다. 우리는 특징이 몇 개 없는 모델만 고려할 것이다 (n=20; m=5; I=0.7). 이를 통해 무슨 일이 일어나는지 시각적으로 쉽게 볼 수 있을 것이다. 우리는 선형 모델과 서로 다른 특징 희소성을 가진 데이터로 훈련된 여러 ReLU 출력 모델을 고려한다.

![[TMS_11.png]]
ReLU Output Model에서, synthetic data의 sparsity $S$를 늘렸을 때, 점점 늘어나는 모습

Superposition은 모델이 더 많은 feature를 표현하는 것을 가능하게 한다. \
그중에서도 가장 중요한 feature는 초기에는 건드려지지 않는 모습. (아마도 antipodal pair들일 것이다.)

Sparsity가 증가하면, 모델은 모든 feature를 superposition에 넣게 되며, 더 많은 정보를 저장할 수 있게 된다. \
이 지점에서 positive interference와 negative biases가 발생하게 되는 것에 주목하자.

## Mathematical understanding

# Superposition as a Phase Change

> The results in the previous section seem to suggest that there are three outcomes for a feature when we train a model: **(1) the feature may simply not be learned**; **(2) the feature may be learned, and represented in superposition**; or **(3) the model may represent a feature with a dedicated dimension**. The transitions between these three outcomes seem sharp. Possibly, there's some kind of phase change.
> 
> 앞서의 결과는 우리가 모델을 훈련할 때 반복되는 세 가지 결과를 제시하는 것 같다: **(1) 특징이 단순히 학습되지 않을 수 있다**; **(2) 특징이 학습되고 중첩으로 표현될 수 있다**; 또는 **(3) 모델이 특정 차원으로 특징을 표현할 수 있다**. 이 세 가지 결과 간의 전환은 뚜렷해 보인다. 아마도 어떤 형태의 위상 변화가 있을 것이다.

이걸 더 잘 이해하는 한 가지 방법은 물리학에서 "phase diagram" 같은 것이 있는지 탐구하여 특정 특징이 어느 이러한 영역에 있을 것으로 예상되는지를 파악하는 것이다. \
이전 실험에서 이러한 힌트를 볼 수 있지만, 많은 특징이 동시에 변하고 상호작용 효과가 있을 수 있기 때문에 실제로 무슨 일이 일어나고 있는지를 분리하기 어렵다. \
결과적으로, 효과를 더 잘 분리하기 위해 다음과 같은 실험을 설정했다.

초기 실험으로, ReLU output 모델 $(ReLU(W^TWx-b))$에서, 2개의 feature가 있고, 1개의 hidden layer 차원이 있는 모델을 고려한다. ($n=2, m=1)$ \
첫 번째 특징의 중요도를 1.0으로 하고, 한 축에서 2번째 "추가" feature의 중요도를 0.1에서 10까지 변화시키고, 다른 축에서는 모든 feature의 sparsity를 1.0에서 0.01로 변화시킨다. \
이후 우리는 2번째 "추가" feature가 학습되지 않았는지, superposition으로 학습되었는지, 또는 orthogonal하게 표현되었는지를 나타내는 플롯을 작성한다. \
노이즈를 줄이기 위해 각 지점에 대해 10개의 모델을 훈련하고 결과를 평균하여 최고 손실을 가진 모델은 제외한다.

이 결과를 이론적인 "toy model of the toy model"과 비교할 수 있다. \
여기서는 중요도와 sparsity를 함수로 하여 다양한 가중치 구성에서 손실에 대한 닫힌 형태의 솔루션을 얻을 수 있다. \
1차원에 2개의 특징을 저장하는 자연스러운 방법에는 세 가지가 있다: 
1. 추가 특징을 버리기 위해 무시하고, 
2. 첫 번째 특징을 버림으로써 추가 특징에 전용 차원을 부여하고, 
3. 특징을 중첩으로 저장하되 두 특징의 조합을 동시에 표현할 능력을 잃는 것이다. 

![[TMS_12.png]]
우리는 이 마지막 솔루션을 "**antipodal**"이라고 부른다. 왜냐하면 두 basis 벡터가 정반대 방향으로 매핑되기 때문이다. 이들 솔루션에 대한 손실을 분석적으로 결정할 수 있는 것으로 나타난다.

![[TMS_13.png|400]]

예상대로, 희소성은 중첩이 발생하는 데 필요하지만, 우리는 이것이 상대적 특징 중요성과 흥미로운 방식으로 상호작용한다는 점을 알 수 있다. \
가장 흥미로운 것은, 경험적 및 이론적 다이어그램 모두에서 실제로 상전이가 관찰된다는 것이다. \
최적의 가중치 구성은 크기와 중첩에서 불연속적으로 변화한다. \
(이론 모델에서는 첫 번째 차수 상전이가 있음을 해석적으로 확인할 수 있으며, 이는 함수 간의 교차가 발생해 최적 손실의 도함수에 불연속성을 만든다는 것을 나타낸다.)

그렇다면, 두 차원에서 세 가지 특징을 포함하는 동일한 질문을 할 수 있을 것이다. \
이 문제는 여전히 우리가 연구할 수 있는 단일 "추가 feature" (이제 세 번째 특징)를 갖고 있으며, 다른 두 특징과의 상대적 중요성을 조정하고 sparsity를 변화시킬 때 어떤 일이 발생하는지 질문한다. \
이론 모델을 위해 이제 네 가지 자연스러운 해결책을 고려한다. \
우리는 "어떤 특징 방향을 무시했는가?"라는 질문을 통해 해결책을 설명할 수 있다. 
1. 추가 특징을 단순히 나타내지 않을 수 있다  $(W\perp[0,0,1])$ 로 표현한다. 
2. 또는 다른 특징 중 하나를 무시할 수도 있다 $(W\perp[1,0,0])$.
3. "추가 feature"을 다른 하나와 반대 쌍으로 배치 $(W\perp[0,1,1])$ 하거나, 
4. 다른 두 특징을 superposition시키고 추가 feature에 dedicated dimension을 부여할 수 있다 $(W\perp[1,1,0])$. 

모든 특징을 공동 중첩으로 두는 마지막 해결책인 $W\perp[1,1,1]$ 에 대해서는 고려하지 않는다.
![[TMS_14.png]]

![[TMS_15.png|400]]

> [!tip] [[Why does it happen?]]

이 diagram들은 특성을 인코딩하기 위한 서로 다른 전략 간의 실제로 phase change가 존재한다는 것을 제안한다. 그러나 다음 섹션에서는 이 예비적인 관점이 포착하지 못하는 더 복잡한 구조가 있다는 것을 볼 것이다.

이번 섹션을 한마디로 요약하자면, 다음과 같이 나타낼 수 있다. \
**Superposition이 모델이 추가 특성(extra feature)을 나타낼 수 있게 하고,** \
**추가 특성의 수가 희소성(sparsity)이 증가함에 따라 증가한다.**

# The Geometry of Superposition

> In this section, we'll investigate this relationship in more detail, discovering an unexpected geometric story: **features seem to organize themselves into geometric structures** such as pentagons and tetrahedrons! In some ways, the structure described in this section seems **"too elegant to be true"** and we think there's a good chance it's at least **partly idiosyncratic** to the toy model we're investigating. But it seems worth investigating because if anything about this generalizes to real models, it may give us a lot of leverage in understanding their representations.
> 
> 이 섹션에서는 이 관계를 더 자세히 조사하여 예상치 못한 기하학적 이야기를 발견할 것이다: Feature들이 오각형과 사면체와 같은 **기하학적 구조로 스스로 조직**되는 것처럼 보인다! 어떤 면에서는, 이 섹션에서 설명하는 구조가 **"너무 우아해서 사실일 리가 없다"** 고 느껴지며, 우리가 조사하고 있는 장난감 모델에 적어도 **부분적으로는 특이한(idioyncratic) 현상**일 가능성이 높다고 생각한다. 그러나 이 부분이 실제 모델로 일반화된다면 이들의 representation을 이해하는 데 많은 도움이 될 수 있기 때문에 조사할 가치가 있어 보인다.

모든 feature가 동일한 **uniform superposition**을 조사하는 것부터 시작해보자: 모든 feature가 **independent**하고, **동일하게 중요**하며, **동일하게 sparse**하다. 놀랍게도 uniform superposition은 균일 다면체의 기하학과 놀라운 연관성을 가지고 있다! 이후에는 feature가 identical하지 않은 non-uniform superposition을 조사할 것이다. 이 부분은 적어도 어느 정도까지는 uniform superposition의 변형으로 이해될 수 있는 것으로 보인다.
## Uniform Superposition

우리는 특성 희소성, 즉 $S$ 를 변화시킬 때 어떤 일이 발생하는지 이해하고자 한다.

Feature의 수를 측정하기 위해 Frobenius norm인 $||W||_F^2$ 를 살펴본다. \
- 만약 특성이 표현된다면, $||W_i||^2 \backsimeq 1$ 이 될 것이고,
- 그렇지 않다면 $||W_i||^2 \backsimeq 0$ 이 될 것이다.
 이는 대략적으로 모델이 표현하는 feature의 수가 된다. \
 이 norm은 basis independent해서, feature basis가 previleged되지 않은 dense한($S=0$) 영역에서도 잘 작동한다.

"Dimensions per feature"을 의미하는 $D^* = \frac{m}{||W||^2_F}$ 을 그래프로 나타내면, 다음과 같다.
![[TMS_16.png]]

흥미로운 점은, 이 그래프가 1과 1/2에서 "sticky"하다는 것이다. \
살펴보면, "sticky point"는 feature가 "**antipodal pairs**"로 구성된 정밀한 기하학적 배열에 해당하는 것으로 보이며, 각 쌍은 서로의 정확한 음수로, 두 feature가 각 hidden dimension에 공존할 수 있게 한다. \
Antipodal pairs가 매우 효과적이라서 모델이 넓은 sparsity 영역에서 이를 선호하여 사용한다고 볼 수 있을 것이다.

## Feature Dimensionality

앞선 그래프에서 모델이 어떤 의미에서 "half a dimension per feature(antipodal pair)"를 가지는 sticky한 영역이 존재한다는 것을 살펴봤다. \
이는 모델이 표현하는 feature의 평균 통계적 특성이지만, 흥미로운 무언가를 암시하는 것 같다. \
특정 feature가 얻는 "fraction of a dimension"을 이해할 방법이 있을까?

이에 대해 $i$ 번째 feature의 dimensionality를 $D_i$로 정의한다.
$$

D_i = \frac{\|W_i\|^2}{\sum_j (\hat{W}_i \cdot W_j)^2}

$$
이를 시각화 해보면, \
![[TMS_17.png|500]] \
이렇게 표현해 볼 수 있다.

실제로 앞선 예제들의 dimensionality를 구해 보면, 다음과 같이 구할 수 있다. \
![[TMS_19.png|500]] \
Feature가 두 쌍의 antipodal pairs를 가지게 되고, 각각의 feature는 0.5 차원을 담당하게 된다.

![[TMS_18.png|500]] \
모든 feature가 interfere를 가지는 superposition 형태로, 각각의 feature는 0.2차원을 담당하게 된다.

앞서 언급했던 $D^*$ 그래프 위에 새로운 산점도 그래프를 그려보자. \
각 sparsity 수준에서, 모델의 각 feature에 대한 개별 특성의 dimensionality를 구할 것이다. \
이때, dimensionality는 특정 비율에서 군집을 이루므로, 그에 맞는 선을 그린다. \
각 feature를 node로 하고, edge 가중치는 내적 특성 임베딩 벡터의 절대값에 기반한다. (특성들이 직교하지 않으면 연결된다) \
![[TMS_20.png]]

특정 dimensionality에서 점들이 모이는 이유가 무엇일까? \
곧 모델이 특정 가중치 기하학을 생성하고 다양한 구성 사이를 "**점프**"하는 경향이 있음을 알 수 있다.

이전 섹션에서는 phase change를 superposition의 이론으로 발전시켰다. \
하지만 이 그래프에서 **0(feature not learned)과 1(dedicated dimension) 사이의 모든 것은 superposition**이다. \
Superposition은 특징들이 분수 차원을 가질 때 발생하는 것이다. 즉, 초월은 단순한 하나의 개념이 아니다!

이것을 우리가 원래 이해했던 위상 변화와 어떻게 연결할 수 있을까? \
우리는 종종 물이 세 가지 상: 얼음, 물, 증기만 가지고 있다고 생각한다. \
하지만 이는 단순화된 표현이다. 실제로는 여러 가지 얼음의 위상이 존재하며, 이는 종종 서로 다른 결정 구조(예: 육각형과 입방체 얼음)를 가진다. \
모호하게나마, 신경망의 feature도 "superposition"이라는 일반 범주 내에서 많은 **다른** 위상을 가진 것처럼 보인다.

## Why these Geometric structures?

이전 다이어그램에서 우리는 차원의 $\frac{3}{4}$(tetrahedron), $\frac32$(triange), $\frac12$(antipodal pair), $\frac25$(pentagon), $\frac38$(square antiprism), 0(not learned)에 해당하는 뚜렷한 선들이 존재함을 발견했다.
기본 특징이 밀집 영역에서 다른 방향들과 구별되지 않는다는 사실이 없다면 1(dedicated dimension) 선도 있을 것이라 믿는다.

이러한 여러 구성 중 몇 가지는 유명한 Thomson problem의 해로 떠오를 수 있다. \
이전에 보았듯이, 우리의 모델은 일반화된 톰슨 문제를 해결하는 것으로 이해될 수 있는 매우 실질적인 의미가 있다. \
모델이 특징을 표현하기로 선택할 때, 그 특징은 n차원 구상에 점으로 임베딩된다.


> [!Tip] Thomson Problem?
> ![[TMS_21.png|300]]
> 
> 쿨롱 법칙에 의해 주어진 힘으로 서로 밀어내는 단위구의 표면에 구속된 N 전자의 최소 정전기 위치 에너지 구성을 결정하는 문제 \
> 서로 밀어내는 입자들이 구의 표면에서 가능한 한 멀리 떨어져 있으려는 최적의 위치를 찾는 문제. 이 배치가 에너지를 최소화하는 구성이라고 할 수 있다.

추가적으로, 톰슨 솔루션에 대한 선들이 uniform polyhedra(다면체)(예: 사면체)에 해당하지만, non-uniform solution을 기대하는 곳에서는 분할하는 선이 있는 것으로 보인다. \
Uniform polyhedra에서는 모든 정점이 동일한 기하학적 특성을 가지므로, feature를 임베드할 경우 각 feature는 동일한 차원을 가지게 된다. 그러나 non-uniform polyhedra로 특징을 임베드하면, 서로 다른 특징들이 서로에게 간섭을 덜 하거나 더 많이 하게 된다.

특히, 많은 톰슨 솔루션은 더 작은 uniform polyhedra의 orthogonal subspace에 두 개의 다면체를 임베드하는 tegum product(곱)으로 이해할 수 있다. (Feature geometry의 이전 그래프 시각화에서는 두 개의 서브그래프가 다른 tegum 인자가 있는 경우에만 분리되어 있다.) \
그 결과, 이들의 차원은 실제로 기본 인자 균일 다면체와 일치할 것으로 예상된다. \
이것은 사실은 더 높은 차원의 문제를 연구하고 있음에도 불구하고 3D 톰슨 문제 솔루션을 관찰하는 가능한 이유를 제시한다. 많은 3D 톰슨 솔루션이 2D 및 1D 솔루션의 tegum product인 것처럼, 아마도 high-dimension solution은 1D, 2D, 그리고 3D solution의 tegum product일 것이다.

## Aside: Polytopes and Low-Rank Matrices

이 시점에서, polytope(다포체: 모든 차원에서 정의된 고차원 구조)와 대칭적이고 양의 정부호이며 저차원 행렬 간에는 상관관계가 있다는 것을 명백히 하는 것이 중요하다.

어떤 면에서 이 상관관계는 자명한데, rank 가 $m$ 인 $n \times n$ 행렬 $W^TW$ 이 있는 경우, $W$ 는 $n \times m$ 행렬이다. \
우리는 $W$의 열을 $m$-차원 공간의 점으로 해석할 수 있다. \
여기서 흥미로운 점은 이것이 기하학에 의해 "$W$"가 유도된다는 것을 명확히 한다는 것이다. \
특히, 비대각 성분이 점의 기하학에 의해 유도되는 방식을 볼 수 있다.

또 다른 방법으로 말하자면, polytope와 superposition 전략 사이에는 정확한 상관관계가 있다. \
예를 들어, 2차원 공간에서 세 개의 특징을 중첩시키는 모든 전략은 삼각형에 해당하고, 모든 삼각형은 그러한 전략에 해당한다. \
이러한 관점에서 볼 때, 만약 우리가 세 개의 동일하게 중요하고 sparse한 feature를 가진다면, 최적의 전략이 정삼각형인 것은 당연할 것이다.

## Non-Uniform Superposition

지금까지는 동일한 중요성, 동일한 sparsity, 그리고 독립적인 feature를 가질 때만을 고려했다. \
모델은 필연적으로 Thomsom problem을 해결하려 했을 것이다. \
모든 feature가 같다면, uniform polyhedra가 가장 적은 loss를 가질 것을 solution으로 제공하려 했을 것이기 때문이다.

이 section에서는 feature들이 uniform 하지 않는, non-uniform superposition에 대해서 다룰 예정이다. \
중요성과 sparsity가 달라지고, 독립적으로 작용하지 않는 correlated 구조를 가질 것이다. \
이렇게 변화되는 요소들은 이전 섹션의 geometry를 비틀게 될 것이다.

실제로 신경망에서 발생하는 superposition은 non-uniform일 것으로 예상되기에, 이에 대한 이해는 중요하다. \
하지만, 기존의 복잡한 이론의 geometry를 non-uniform superposition에 그대로 적용시키는 것은 어렵다. \
그래서 이 섹션의 목표를 다음과 같은 현상을 강조하여 설명하고자 한다:

- **Features varying in importance or sparsity**

	**중요성이나 희소성이 변동하는 특징**들은 불균형이 형성될 때 다면체의 부드러운 변형을 발생시킨다. \
	이렇게 변형이 진행되다가 임계 파손점에 도달하면 다른 다면체로 전환된다.
	
- **Correlated features**

	**Correlated features**들은 종종 서로 다른 tegum 인자에서 형성되며 직교하는 것을 선호한다. \
	결과적으로 직교하는 지역 기저를 형성할 수 있다. \
	직교할 수 없는 경우에는 나란히 있는 것을 선호한다. \
	어떤 경우에는 상관된 특징들이 단일 특징으로 통합된다. (superposition-like vs PCA-like)
	
- **Anti-correlated features**

	**Anti-correlated feature**끼리는 superposition이 필요한 경우, 같은 tegum 인자에 있는 것을 선호한다. \
	그들은 이상적으로 반대 위치에서 negative interfere을 가지는 것을 선호한다.

### Perturbing a single feature

Non-uniform superposition을 확인하는 가장 간단한 방법은 다른 feature들을 uniform하게 고정하고, 하나의 feature만 변화시키는 것이다. \
실험으로, $n=5$ , $m=2$ 인 상황을 가정해보자. \
Uniform case에서는, 중요도 $I=1$ 과 sparsity $1-S=0.05$ 를 가지게 되어 pentagon의 형태를 가지게 된다. \
그러나, 1개의 feature에 대해 sparsity를 변화시키면, 정오각형이 새로운 값을 고려하기 위해 stretch되는 것을 볼 수 있다. \
해당 feature의 sparsity를 낮추게 되면, 더 자주 활성화되어(yellow), 다른 feature들이 해당 feature로부터 밀려나면서 더 많은 공간을 가지게 된다. \
반대로, sparse하게 만들면, 덜 자주 활성화되어(blue), 덜 공간을 차지하고 다른 feature들이 그쪽으로 밀려난다.

![[TMS_22.png]]

우리가 충분히 sparse하게 만들면, phase change가 발생하여 pentagon에서 digon로 붕괴되며, 밀집하지 않은 점은 zero로 위치하게 된다. \
이 phase change는 두 가지 다른 기하학이 교차하는 loss curve에 해당한다. \

![[TMS_23.png]]


> [!Tips] Experiment
> ![[TMS_24.png]]
> 
> 다음은 한 feature를 


