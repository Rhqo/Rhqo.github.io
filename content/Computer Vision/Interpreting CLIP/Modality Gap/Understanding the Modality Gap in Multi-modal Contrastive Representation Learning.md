
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

Modality gap이 존재하기 위해서는 encoder의 embedding이 전체 embedding space의 하위 영역(subregion)에 집중되어야 한다. \
그렇지 않으면 서로 다른 encoder에서의 embedding이 겹칠 것이다. \
이 점에서 영감을 받아, cone effect로 인해 임의의 모델 initialization에서 이미 modality gap이 발생함을 보여주는 것으로 조사를 시작한다.

sdfasdf
## The Narrow Cone of Embeddings
### How narrow is the cone in 512-dim representation space?

## The effects of non-linear activation on cone effect
### Design
### Results
## Different random initializations create different cones
# Theoretical analysis
# Contrastive learning preserves modality gap