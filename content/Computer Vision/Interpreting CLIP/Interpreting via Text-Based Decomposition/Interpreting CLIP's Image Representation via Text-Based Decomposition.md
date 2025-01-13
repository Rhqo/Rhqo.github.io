
> [!Abstract] Abstract
> We investigate the CLIP image encoder by analyzing how individual model components affect the final representation. We decompose the image representation as a sum across individual **image patches**, **model layers**, and **attention heads**, and use CLIP’s text representation to interpret the summands. Interpreting the attention heads, we characterize each head’s role by automatically finding text representations that span its output space, which reveals property-specific roles for many heads (e.g. location or shape). Next, interpreting the image patches, we uncover an emergent spatial localization within CLIP. Finally, we use this understanding to remove spurious features from CLIP and to create a strong zero-shot image segmenter. Our results indicate that a scalable understanding of transformer models is attainable and can be used to repair and improve models.
> 
> ---
> 우리는 개별 모델 구성 요소가 최종 표현에 어떤 영향을 미치는지 분석하여 CLIP 이미지 인코더를 조사합니다. 이미지 표현을 개별 **image patches**, **model layers**, **attention head**에 걸쳐 합으로 분해하고, CLIP의 텍스트 표현을 사용하여 요약을 해석합니다. 주의 헤드를 해석하여 출력 공간에 걸쳐 있는 텍스트 표현을 자동으로 찾아 각 헤드의 역할을 특성화하고, 이는 많은 헤드(예: 위치 또는 모양)에 대한 속성별 역할을 드러냅니다. 다음으로, 이미지 패치를 해석하여 CLIP 내에서 새로운 공간적 위치를 발견합니다. 마지막으로, 이러한 이해를 바탕으로 CLIP에서 가짜 특징을 제거하고 강력한 제로샷 이미지 세그먼트를 생성합니다. 우리의 결과는 트랜스포머 모델에 대한 확장 가능한 이해가 가능하며 모델을 복구하고 개선하는 데 사용할 수 있음을 나타냅니다.


# 1. Introduction

> To better understand CLIP, we design methods to study its internal structure, focusing on CLIP-ViT. Our methods leverage several aspects of CLIP-ViT’s architecture:
> First, the architecture uses **residual connections**, so the output is a **sum of individual layer outputs.**
> Moreover, it uses **attention**, so the output is also a **sum across individual locations** in the image.
> Finally, the representation lives in a **joint vision-language space**, so we can **label its directions with text**.
> We use these properties to decompose the representation into text-explainable directions that are attributed to specific attention heads and image locations.


Residual structure를 사용하여 어떤 레이어가 출력에 상당한 직접 효과를 미치는지 조사한다. \
마지막 4개의 attention layer를 제외한 모든 layer를 가리는 경우 CLIP의 제로샷 분류 정확도에 거의 영향을 미치지 않음을 발견했다.

→ CLIP 이미지 표현이 주로 이러한 후기의 attention layer에 의해 구성된다는 결론
### TEXTSPAN
그래서 후기의 4개의 attention layer에 대해서 조사한다. \
각 attention layer의 head의 basis를 찾는 알고리즘 **TEXTSPAN**을 제안한다. \
각 basis는 text caption으로 labeling된다. \
결과로 나오는 basis는 각 헤드의 specialized된 역할을 드러낸다. \
ex) 한 head의 상위 3개 basis 방향은 "반원형 아치", "이등변 삼각형", "타원"으로 나타나며 이는 형태에 전문화되었음을 알 수 있다.

이는 두가지 방법으로 응용 가능하다.

- 첫째, 불필요한 신호와 관련된 head를 제거함으로써 불필요한 상관관계를 줄일 수 있다.
- 둘째, 특성별 역할을 가진 head의 표현을 사용하여 해당 특성에 따른 이미지를 검색할 수 있다.

→ 색상, 위치 및 질감과 같은 발견된 유사성의 감각을 기반으로 한 검색을 수행하는 데 이를 사용

### Spatial Structure
다음으로 우리는 attention layer가 제공하는 **spatial structure**를 활용한다. \
각 attention head의 출력은 이미지 위치에 대한 weighted sum이므로, 이러한 위치에 따라 출력을 분해할 수 있다. \
주어진 텍스트 방향을 따라 각 위치가 얼마나 기여하는지를 시각화 가능하다. \
이는 기존 CLIP 기반 제로샷 방법보다 성능이 우수한 제로샷 이미지 segmentation을 제공한다.
### TEXTSPAN + Spatial Structure
우리는 **TEXTSPAN**에서 얻은 텍스트 기저와 함께 **Spatial Structure**를 고려한다. \
기저의 각 방향에 대해 spatial decomposition는 해당 기저 방향에 영향을 미치는 이미지 영역을 강조한다. \
우리는 이를 그림 1(다)에서 시각화하며, 이는 우리의 텍스트 레이블을 검증하는 것으로 나타났습니다: 예를 들어, 삼각형이 있는 영역은 이등변 삼각형으로 label된 방향에 주요 기여자이다.

요약하자면, 우리는 CLIP의 이미지 표현을 개별 attention head 및 이미지 위치와 관련된 텍스트 해석 가능 요소로 분해하여 해석한다.

이를 통해 Property specific heads와 emergent localization을 발견하고, 우리의 발견을 사용하여 가짜 신호를 줄이고 zero-shot segmentation을 개선하여, 이해가 다운스트림 성능을 향상시킬 수 있음을 보여준다.

# 3. Decomposing CLIP image representation into layers

## 3.1 CLIP-ViT Preliminaries
### Contrastive pre-training

Text encoder $M_{text}$ 와 Image encoder $M_{image}$ 를 사용, 동일한 latent space에 매핑되어 cosine similarity를 통해 텍스트 간의 유사성을 측정할 수 있다. \

$$
\begin{equation}\text{sim}(I,t) = \frac{\langle M_{\text{image}}(I), M_{\text{text}}(t) \rangle}{\|M_{\text{image}}(I)\|_2 \|M_{\text{text}}(t)\|_2}\end{equation}
$$

일련의 이미지와 해당 텍스트 설명 $\{(I_i, t_i)\}_{i \in \{1,...,k\}}$*가 주어졌을 때, CLIP은 이미지 표현 $M_{{image}}(I_i)$*와 해당 텍스트 표현 $M_{\text{text}}(t_i)$의 유사성을 극대화하고 배치 내 $i \neq j$에 대해 $\text{sim}(I_i,t_j)$를 최소화하도록 훈련된다.

### Zero-shot classification  

CLIP은 제로샷 이미지 분류에 사용할 수 있다. 고정된 클래스 집합이 주어졌을 때, 각 클래스 이름(예: "치와와")은 고정 템플릿(예: "An image of a {class}")에 매핑되어 CLIP 텍스트 인코더에 의해 인코딩된다. 주어진 이미지에 대한 예측은 해당 텍스트 설명과 가장 높은 유사성을 가진 클래스이다.
### CLIP image representation

CLIP의 이미지 표현을 계산하기 위해 여러 아키텍처가 제안되었다. 우리는 ViT를 백본으로 사용하는 변형에 집중한다. 여기서 비전 변환기(ViT)가 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$ 에 적용되어 $d$-차원 표현 $\text{ViT}(I)$ 를 얻는다. CLIP 이미지 표현 $M_{\text{image}}(I)$ 는 이 출력을 공동 시각-언어 공간에서 $d'$-차원 표현으로 선형 투영한 것이다. 공식적으로 투영 행렬을 $P \in R^{d^′\times d}$로 나타낸다:

$$ 
\begin{equation}M_{\text{image}}(I) = P \text{ViT}(I)\end{equation}
$$

$\text{ViT}$와 projection 행렬 $P$의 매개변수는 훈련 중에 학습된다.
### ViT architecture

ViT는 각 레이어가 multi-head self-attention(MSA)와 MLP 블록으로 구성된, $L$개의 레이어로 이루어진 residual 네트워크이다.

입력 $I$ 는 먼저 $N$ 개의 겹치지 않는 이미지 패치로 분할된다. 패치는 $N$ 개의 $d$ -차원 벡터로 선형적으로 투영되며, 위치 임베딩이 추가되어 이미지 토큰 $\{z^0_i\}_{i \in \{1,...,N\}}$ 를 생성한다. 추가 학습된 토큰 $z^0_0 \in \mathbb{R}^d$ , 즉 클래스 토큰도 포함되며, 이후 출력 토큰으로 사용된다.

형식적으로, 토큰 $z_0^0, z_1^0, ..., z_N^0$ 가 열로 구성된 행렬 $Z^0 \in R^{d\times(N+1)}$ 은 residual stream의 initial state를 구성한다. 이는 다음과 같이 두 개의 residual step에 따라 $L$ 번 반복적으로 업데이트된다.
$$

\begin{equation}\hat{Z}^l = \text{MSA}_l(Z^{l-1}) + Z^{l-1}, \quad Z^l = \text{MLP}_l(\hat{Z}^l) + \hat{Z}^l\end{equation}

$$

Residual stream $Z^l$ 의 첫번째 열을 클래스 토큰에 해당하는 $[Z^l]_{cls}$ 로 표시한다. 따라서, ViT의 출력은 $[Z^l]_{cls}$ 이다.

## 3.2 Decomposition into Attention heads

ViT의 residual 구조는 모델의 개별 layer의 직접적인 기여의 합으로 출력을 표현할 수 있게 해준다. \
Image representation인 $M_{image}(I)$ 가 ViT output의 linear projection이라는 것을 생각해보면, 식 (3)을 통해 다음과 같이 쓸 수 있다.  

$$

\begin{equation}M_{\text{image}}(I) = P_{\text{ViT}}(I)= P[Z^0]_{cls} + \underbrace{\sum_{l=1}^{L} P\left[{\text{MSA}^l}(Z_{l-1})\right]_{cls}}_{\text{MSA terms}} + \underbrace{\sum_{l=1}^{L} P\left[\text{MLP}^l(\hat{Z}_l)\right]_{cls}}_{\text{MLP terms}} \end{equation}

$$

식 (4)는 Image representation을 MLP, MSA, input class token의 direct contribution들로 분해할 수 있으며, 이를 통해 각 항목을 개별적으로 분석할 수 있다. 여기서는 한 layer의 출력이 downstream 계층에 미치는 간접 효과를 무시한다. 이 분해(및 추가 분해)를 사용하여 다음 섹션에서 CLIP의 표현을 분석한다.

### Evaluating the direct contribution of layers

식 (4)에서 어떤 구성 요소가 최종 이미지 표현에 유의미한 영향을 미치는지 연구했고, 대다수의 직접 효과가 후반 attention layer에서 온다는 것을 발견했다.

구성 요소(또는 구성 요소 집합)의 직접 효과를 연구하기 위해, 평균 제거(mean-ablation) 기법을 사용하여 구성 요소를 이미지 데이터셋의 평균 값으로 대체한다. 구체적으로, 제거 전후의 classification 작업에서 zero-shot 정확도의 감소를 측정한다. 더 큰 직접 효과를 가진 구성 요소는 더 큰 정확도 감소를 가져야 한다.

실험에서는, ImageNet validation set에서 각 구성 요소의 평균을 계산하고 ImageNet classification의 정확도의 감소를 평가한다. LAION-2B에서 훈련된 OpenCLIP ViT-H-14, L-14, B-16 모델을 분석한다.
### MLPs have a negligible direct effect

![[content/Computer Vision/Interpreting CLIP/Interpreting via Text-Based Decomposition/ICIRTBD_0.png]]

표 1은 모든 MLP를 동시에 평균 제거한 결과를 보여준다. MLP는 이미지 표현에 유의미한 직접 효과가 없으며, 모두 제거해도 정확도가 1%-3%만 떨어지게 된다.
### Only the last MSAs have a significant direct effect

![[content/Computer Vision/Interpreting CLIP/Interpreting via Text-Based Decomposition/ICIRTBD_1.png]]

다음으로 다양한 MSA 계층의 직접 효과를 평가한다. 이를 위해, 우리는 어떤 계층 $l$ 까지 모든 MSA 계층을 평균 제거합니다. 그림 2는 결과를 보여준다: 초기 MSA 계층(마지막 4개까지)을 제거해도 정확도에 큰 변화가 없지만, 마지막 MSA를 평균 제거하면 성능이 급격히 감소하게 된다.

요약하자면, 출력에 대한 직접 효과는 마지막 4개의 MSA 계층에 집중되어 있다. 따라서 이후 분석에서는 이러한 계층에만 초점을 맞추고 MLP와 초기 MSA 계층은 무시한다.

## 3.3 Fine-Grained Decomposition into Heads and Positions

다음 두 섹션에서 사용될 MSA 블록의 더 세분화된 분해를 제시한다. 클래스 토큰의 출력을 중심으로 하며, 이는 식 (4)에서 나타나는 유일한 항이다. Elhage et al.(2021)을 따르며 MSA 출력을 H개의 독립적인 attention 헤드와 N개의 입력 토큰의 합으로 나타낸다:


$$

\begin{equation}

\left[\text{MSA}(Z_{l-1})\right]_{cls} = \sum_{h=1}^{H} \sum_{i=0}^{N} x^{l,h}_i, \quad x^{l,h}_i = \alpha^{l,h}_i W^{l,h}_{VO}\approx^{l-1}_i

\end{equation}

$$


$W^{l,h}_{VO} \in \R^{d\times d}$ 는 transition 행렬이며, $\alpha^{l,h}_i \in \R$ 는 class 토큰에서 i번째 토큰까지의 attention 가중치이다. $(\sum^N_{i=0}\alpha^{l,h}_i = 1)$ \
따라서, MSA 출력은 개별 head와 토큰의 직접 효과로 분해될 수 있다. \
식 (5)의 MSA 출력 정의를 식 (4)의 MSA 항에 대입하면 다음과 같이 나타낼 수 있다:
$$

\begin{equation}\sum^L_{l=1}P\left[\text{MSA}^l(Z^{l-1})\right]_{cls} = \sum^L_{l=1}\sum^H_{h=1}\sum^N_{i=0}c_{i,l,h}, \;\;c_{i,l,h} = Px^{l,h}_i\end{equation}

$$

모든 attention block의 총 직접 효과는 텐서 $c$ 를 모든 차원에서 수축한 결과이다. 일부 차원에서만 수축함으로써 다양한 유용한 방식으로 효과를 분해할 수 있다.

예를 들어, 공간 차원 $i$ 를 따라 수축하여 각 헤드에 대한 기여를 얻을 수 있다: $c^{l,h}_{head} = \sum^N_{i=0}c_{i,l,h}$ . 대신, layer와 head에 따라 수축해서 각 이미지 토큰으로부터 기여를 얻을 수 있다: $c^i_{token} = \sum^L_{l=1}\sum^H_{h=1}c_{i,l,h}$ .

$c_{i,l,h}$ , $c^{l,h}_{head}$ , $c^i_{token}$ 은 모두 $d'$ - 차원의 text-image representation space에 존재하며, 이를 통해 텍스트를 해석할 수 있다. 예를 들면, 텍스트 설명 $t$ 가 주어졌을 때 $\lang M_{text}(t), c^{l,h}_{head} \rang$ 는 해당 헤드의 출력과 $t$ 의 유사성을 직관적으로 측정한다.

# 4. Decomposition into attention heads

3.2에서 보았듯이, CLIP의 후기의 MSA layer를 이해하는 것에 초점을 맞춘다. 섹션 3.3에서 보았듯이, 개별 attention head로의 분해를 사용하고, 각 헤드의 latent direction에 텍스트 설명으로 레이블을 부여하는 알고리즘을 제시한다.

![[content/Computer Vision/Interpreting CLIP/Interpreting via Text-Based Decomposition/ICIRTBD_2.png]]

![[content/Computer Vision/Interpreting CLIP/Interpreting via Text-Based Decomposition/ICIRTBD_3.png]]

이 레이블링의 예시는 표 2와 그림 4에 나타나 있으며, 64개의 늦은 주의 헤드에 대한 레이블링은 섹션 A.5에서 주어진다.

이러한 labeling은 일부 헤드가 특정 의미 역할을 나타내는 것을 드러낸다. 예를 들어 "숫자 세기"나 "위치" 등에서 많은 latent direction이 그 역할의 다른 측면을 추적합니다. 우리는 이러한 레이블이 붙은 역할을 특성별 이미지 검색 및 불필요한 상관관계 줄이기에 어떻게 활용하는지를 보여준다.
## 4.1 Text-interpretable decomposition into heads

MSA의 출력을 조합 표현 공간의 텍스트 관련 방향으로 분해한다. 두 가지 주요 특성에 의존한다:

- 각 MSA 블록의 출력은 개별 attention 헤드의 기여의 합으로 이루어져 있다.
- 이러한 기여는 joint text-image representation space에 위치하므로 텍스트와 연관될 수 있다.

섹션 3.3에서 이미지 표현의 MSA 항(식 4)이 헤드에 대한 합으로 표현될 수 있다는 것을 기억하자: $\sum_{l,h}c^{l,h}_{head}$ . 헤드의 기여 $c^{l,h}_{head}$ 를 해석하기 위해, 헤드의 출력 변화 대부분을 설명하는 텍스트 설명 집합을 찾는다. (헤드의 “principal component (PC)”

이걸 공식화하기 위해, 입력 이미지 $I_1, ..., I_K$ 및 관련된 헤드 출력 $c_1, ..., c_K$ 를 사용한다. $c_1, ..., c_K$ 가 joint text-image represent space에 존재하는 벡터이기 때문에, 각 text input $t$ 는 $M_{text}(t)$ 의 방향을 정의한다. 텍스트 방향 집합 $T$ 가 주어지면, $\text{Proj}T$는 $\{M_{\text{text}}(t) | t \in T\}$ 의 스팬에 대한 투영을 의미한다. $T$ 에 의해 설명된 분산을 다음과 같이 정의한다:

$$

\begin{equation}V_{explained}(T) = \frac1K\sum^K_{k=1}||Proj_T(c_k - c_{avg})||^2_2, \;\; where\;c_{avg} = \frac1K\sum^K_{k=1}c_k\end{equation}

$$

우리는 각 헤드에 대해 $V_{\text{explained}}(T)$ 를 최대화하는 $m$ 개의 설명 집합 $T$ 를 찾고자 한다. 일반적인 PCA와 달리 이 최적화 문제에 대한 닫힌 형태의 해가 없으므로, 우리는 탐욕적 접근 방식을 취한다.

### Greedy algorithm for descriptive set mining

식 (7)에서 설명된 분산을 대략 최대화하기 위해, 우선 M개의 후보 설명 $\{t_i\}_{i=1}^{M}$ 의 대규모 풀에서 탐욕적으로 선택하여 집합 $T$ 를 얻는다.

![[content/Computer Vision/Interpreting CLIP/Interpreting via Text-Based Decomposition/ICIRTBD_4.png]]