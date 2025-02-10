---
title: 
tags:
  - section
---
# Understanding the Modality Gap in Multi-modal Contrastive Representation Learning

> [!Abstract] Abstract
> We present modality gap, an intriguing geometric phenomenon of the representation space of multi-modal models. Specifically, we show that different data modalities (e.g. images and texts) are embedded at arm’s length in their shared representation in multi-modal models such as CLIP. Our systematic analysis demonstrates that this gap is caused by a combination of model initialization and contrastive learning optimization. In model initialization, we show empirically and theoretically that the representation of a common deep neural network is restricted to a narrow cone. As a consequence, in a multi-modal model with two encoders, the representations of the two modalities are clearly apart when the model is initialized. During optimization, contrastive learning keeps the different modalities separated by a certain distance, which is influenced by the temperature parameter in the loss function. Our experiments further demonstrate that varying the modality gap distance has a significant impact in improving the model’s downstream zero-shot classification performance and fairness.
> 
> 우리는 멀티모달 모델의 표현 공간에서 나타나는 흥미로운 기하학적 현상인 "modality gap"을 제시한다. 구체적으로, CLIP과 같은 멀티모달 모델에서 서로 다른 데이터 모달리티(예: 이미지와 텍스트)가 공유된 representation에서 일정한 거리만큼 떨어져 있게 임베딩된다는 것을 보여준다. 체계적인 분석을 통해, 이러한 간격은 모델 initialization과 contrastive learning 최적화의 결합에 의해 발생한다는 것을 밝혀낸다. 모델 초기화 과정에서, 공통 DNN의 표현이 이론적으로나 실험적으로 좁은 원뿔(cone) 안에 제한된다는 것을 보인다. 그 결과, 두 개의 encoder를 가진 멀티모달 모델에서는 initialization 시 두 모달리티의 표현이 명확히 떨어져 있게 된다. 최적화 과정에서, contrastive learning은 loss function에서 temperature parameter에 의해 영향을 받는 일정한 거리만큼 다른 모달리티들을 분리된 상태로 유지한다. 실험을 통해 모달리티 갭 거리를 조정하면, 모델의 downstream zero-shot classification 성능과 fairness을 크게 개선할 수 있다는 것을 추가로 입증한다.

# Introduction

갭을 데이터 분포의 차이나 서로 다른 인코더 아키텍처의 차이에 기인한다고 추론하는 것이 합리적으로 보일 수도 있지만, 이러한 요인이 기본 원인이 아니라는 것을 보여주고자 한다.

이 논문은 모달리티 갭 현상을 세 부분으로 설명한다.
- **DNN 아키텍처의 일반적인 inductive bias가 cone effect를 생성한다.**

	효과적인 임베딩 공간은 사전 훈련된 모델이나 무작위 가중치를 가진 모델에서 좁은 원뿔로 제한된다.

- **서로 다른 random initialization이 서로 다른 embedding cone을 생성한다.**

	다중 모달 모델은 두 개의 인코더로 구성되며, 이들이 무작위 초기화에서 서로 다른 원뿔을 생성하기 때문에 모달리티 격차가 존재한다.

- **Multi-modal model에서 일반적으로 사용되는 contrastive learning objective가 격차를 보존한다.**

이에 따른 본 논문의 목적을 다음과 같이 세 부분으로 볼 수 있겠다.
1. 다양한 데이터 modality와 NN 전반에 걸쳐 modality gap 현상을 실증적으로 보여주는 것
2. 격차가 발생하는 이유를 설명하는 것
3. 격차의 크기가 downstream task에 미치는 영향을 보여주는 것

**Modality gap이 없는 것이 바람직한지는 불확실하기 때문에, gap을 줄이는 방법을 제안하는 것이 목적이 아니다.**

이 논문은 다음과 같은 기여를 한다.

> [!Quote] Contributions
> 1. 우리가 아는 한, 우리는 **처음으로 일반적인 modality gap 현상을 입증**한다. 우리는 이 현상이 텍스트, 자연 이미지, 비디오, 의료 이미지 및 아미노산 서열을 포함하는 광범위한 multi-modal model에 걸쳐 유효함을 보여준다.
> 2. **Downstream task에서 격차를 수정하는 것의 중요한 implication을 입증**한다. 단순히 격차의 거리를 수정함으로써 CLIP의 zero-shot performance와 fairness를 향상시킬 수 있다.
> 3. Modality gap을 설명하기 위해 우리는 포괄적인 **이론적 및 경험적 분석으로 뒷받침된 세 부분의 설명**을 제공한다. 우리의 분석은 또한 cone effect에 대한 새로운 통찰을 제공한다. 우리는 이 효과가 다양한 모달리티와 네트워크 아키텍처에 걸쳐 존재할 뿐만 아니라 무작위 노이즈 입력과 무작위 가중치에서도 나타난다는 것을 보여주며, 이는 이전 연구에서 포착되지 않았다.
> 4. Cone effect를 설명하기 위해 **ReLU non-linearlity을 가진 linear layer들에 의해 유도된 contraction mapping을 수학적으로 특징화**한다. 우리의 이론은 실험과 잘 맞아떨어지며 심층 신경망의 일반적인 inductive bias를 이해하는 데 중요한 통찰을 제공한다.

# The Cone Effect Induces a Modaltiy Gap
## The Narrow Cone of Embeddings
Modality gap이 존재하기 위해서는 encoder의 embedding이 전체 embedding space의 하위 영역(subregion)에 집중되어야 한다. \
그렇지 않으면 서로 다른 encoder에서의 embedding이 겹칠 것이다. \
이 점에서 영감을 받아, cone effect로 인해 임의의 모델 initialization에서 이미 modality gap이 발생함을 보여주는 것으로 조사를 시작한다.

효과적인 임베딩 공간은 훈련된 모델 및 무작위 가중치를 가진 모델에 대해 좁은 원뿔로 제한된다. \
이를 입증하기 위해, 우리는 각각의 pre-train된 모델(ResNet, ViT, Text Transformer)에서 최종 layer로부터  MSCOCO의 5,000개 캡션의 임베딩을 추출한다. \
그런 다음 각 모델 내에서 5,000개의 임베딩 간의 모든 가능한 쌍의 코사인 유사성을 계산한다. \
우리는 평균 cosine similarity(각각 0.56, 0.47, 0.51)과 최소 cosine similarity(0.23, 0.05, 0.01)이 모두 양수임을 발견했다. \
이러한 결과는 임베딩 공간이 좁은 원뿔임을 나타낸다.

문헌에서 원뿔 효과(cone effect)는 언어 모델(예: BERT)의 언어 표현에서 관찰되었다. \
(Representation Degeneration Problem in Training Natural Language Generation Models) \
일반적인 설명으로는, 단어 빈도의 불균형 분포가 최적화에 편향을 가져온다고 한다. \
그러나 우리는 아래 그림과 같이 random weight를 가진 모델에서도 원뿔 효과가 여전히 존재함을 발견했다.

> [!Archive] Figure 2
> ![[MDG_0.png]]
> 훈련 없이 25개의 무작위로 초기화된 모델의 임베딩을 실제 데이터에서 UMAP 시각화. \
> 각 무작위 초기화는 독특하게 다른 원뿔을 형성한다.
> - Real Data: MSCOCO caption의 validation set에서 5,000개의 image-caption pair.
> - Random Noise: 표준 정규 분포에서 발생하는 Gaussian noise를 이미지로, 균일하게 무작위인 정수 시퀀스를 텍스트로 나타낸다.

실제로, random noise의 평균 cosine similarity는 훈련된 모델보다 더 높다. \
예를 들어, 랜덤 초기화된 ResNet의 두 개 임베딩은 평균적으로 거의 완벽한 (0.99) 코사인 유사도를 가진다. \
흥미롭게도, 입력 데이터가 random noise일 때도 원뿔 효과가 여전히 유지되며, 이는 이전 연구에서 제안된 불균형 데이터 분포가 원뿔 효과에 필요하지 않다는 것을 나타낸다. \
이러한 실험들은 원뿔 효과가 이전에 인식된 것보다 심화된 네트워크의 보다 일반적인 inductive bias를 반영한다고 제안한다.
### How narrow is the cone in 512-dim representation space?
Cosine similarity가 0.56이라도 이미 임베딩 공간이 512차원 feature space에서 실제로 극도로 좁은 원뿔을 나타낸다는 것을 분명히 한다. \
Unit hypersphere의 표면적 비율을 고려해 보자. \
2D에서, $\arccos(0.56) = 55.94^\circ$ 이고, 이는 0.56의 cosine similarity가 2D 단위 원에서 "occupy"할 수 있는 면적이 $55.94^\circ / 360^\circ = 15.53\%$ 임을 나타낸다. \
3D에서는 0.56의 cosine similarity가 $\frac{2\pi r^2 (1 - \cos(55.94^\circ))}{4\pi r^2} = 3.34\%$ 의 3D unit sphere를 "occupy"할 수 있다. \
512D에서는 0.56의 cosine similarity가 512D hypersphere의 표면적의 $\frac{1}{2^{512}}$ 미만을 "occupy"할 수 있다. \
이러한 증거들은 효과적인 임베딩 공간이 극도로 좁은 원뿔로 제한되어 있음을 보여준다.
## The effects of non-linear activation on cone effect

### Design
비선형 활성화 함수가 원뿔 효과에 미치는 영향을 연구하기 위해, 우리는 다양한 MLP를 랜덤으로 초기화하고 서로 다른 비선형성을 적용하거나 비선형성을 적용하지 않은 모델들을 사용했다. \
MLP의 입력은 512차 표준 정규 랜덤 벡터이다. \
모든 MLP 선형 계층은 512 × 512로, 가중치와 편향은 각각 $N(0, \frac{1}{512})$에서 랜덤으로 초기화되며, 여기서 $N(\mu, \sigma^2)$는 평균 $\mu$와 분산 $\sigma^2$를 가진 가우시안 분포를 나타낸다.
### Results
결과적으로, 그림 2(b)에서 비선형 활성화가 없는 MLP는 원뿔 효과가 거의 없음을 보여준다. \
그러나 비선형성이 있을 경우, 평균 코사인 유사도는 layer 수가 증가함에 따라 급격히 증가한다. \
예를 들어, 평균 코사인 유사도는 Sigmoid를 가진 2층 MLP에서 0.99에 도달한다. \
이러한 결과는 비선형 활성화 함수가 원뿔 효과에 중요한 역할을 한다는 것을 나타낸다. \
ReLU가 모든 좌표를 non-negative로 만드는 것을 쉽게 확인할 수 있지만, ReLU 이후의 코사인 유사도가 non-negative로 보장된다는 점을 강조한다. \
그러나 그림 2(a)의 3개 모델 중 어떤 것도 임베딩 추출 전의 마지막 층으로 ReLU를 사용하지 않았다. \
또한 모든 3개 모델이 batch norm과 layer norm와 같은 normalization 계층을 아키텍처에 포함하고 있지만, 우리는 여전히 원뿔 효과를 관찰한다. 정규화와 원뿔 효과 간의 연결을 추가로 분석하는 것은 향후 연구의 흥미로운 방향이다.

## Different random initializations create different cones
# Theoretical analysis
# Contrastive learning preserves modality gap