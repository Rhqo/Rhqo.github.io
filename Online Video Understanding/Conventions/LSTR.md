# LSTR: Long Short-Term Transformer for Online Action Detection

## 1. Introduction


## 2. Related Work
### Online Action Detection
- Temporal action localization은 전체 비디오를 관찰하여 행동 인스턴스를 감지합니다.
- Online action detection는 현재까지의 데이터를 기반으로 하는 인과적 처리를 필요로 합니다.
- 주요 접근법:
    - **RED**: 강화 학습 손실을 활용하여 행동을 조기에 인식.
    - **TRN**: 행동 탐지와 예측을 결합하여 더 큰 시간적 문맥을 모델링.
    - **IDN**: 구별 가능한 특징을 학습하고 관련 정보만 축적.
    - **LAP-Net**: 최적의 특징 추출을 위한 적응적 샘플링 제안.
    - **PKD**: 커리큘럼 학습을 통해 오프라인 모델의 지식을 온라인 모델로 전이.
    - **Shou et al.** 및 **StartNet**: 행동 시작의 온라인 탐지(ODAS)에 중점.
    - **WOAD**: 비디오 레벨 라벨을 사용한 약한 감독 학습.
### Temporal / Sequence Modeling
- 전통적인 분석은 잠재 "상태" 변수를 가정해 시계열을 처리.
- 복잡한 데이터에는 직접적인 역사 모델링 필요.
- 초기 행동 인식은 훈련을 위해 휴리스틱 서브 샘플링에 의존.
- **3D ConvNets**: 시공간 특징 모델링을 수행하나 장기적인 시간 상관 관계는 놓침.
- 최근 방법은 많은 정보를 놓칠 수 있는 장기 특징 뱅크 도입.
- 인지 과학은 주의 메커니즘을 통한 장기 의존성 모델링에 대한 설계 통찰 제공.

### Transformers for Action Understanding
- 트랜스포머는 NLP에서 성공을 거두고, 이미지 인식과 객체 탐지 등 비전 작업에 사용.
- 비디오의 시간적 모델링 작업에 적용하여 행동 인식 및 위치 찾기 수행.
- 대부분의 연구는 계산 제한으로 짧은 클립에 초점, 일부는 장기 컨텍스트 모델링 탐구.
- 장단기 정보 통합은 상대적으로 덜 탐구된 영역.

## 3. Long Short-Term Transformer
### 3.1 Overview
Long Short-Term Transformer (LSTR) 는 라이브 스트리밍 비디오에서 과거와 현재의 관찰만을 사용하여 각 프레임에서 수행되는 행동을 식별하는 방법입니다. 미래 정보는 추론 과정에서 접근할 수 없습니다.

![[LSTR_0.png]]

비디오 스트리밍은 시간 $t$에서 $τ$개의 과거 프레임들로 구성된 $I_t$로 표현되며, 여기서 온라인 행동 감지 시스템은 $I_t$를 입력으로 받아 $K + 1$개의 행동 범주 중 하나에 해당하는 $ŷ_t$를 분류합니다. $k = 0$일 때, $t$ 프레임에서 아무런 사건이 발생하지 않을 확률을 나타냅니다. 사전 학습된 특징 추출기(Feature Extractor)가 각 비디오 프레임 $I_t$를 $C$ 차원의 특징 벡터 $f_t$로 변환하여, 이를 $(τ × C)$ 차원의 시간적 시퀀스로 만듭니다.

LSTR의 구조는 인코더-디코더 방식으로 이루어져 있습니다. 멀리 있는 과거 프레임들의 특징 벡터는 장기 메모리에 저장되고, 최근 프레임들의 특징은 단기 메모리에 저장됩니다. LSTR 인코더는 장기 메모리의 특징을 압축하여 인코딩된 잠재 표현을 생성하고, LSTR 디코더는 이 인코딩된 장기 메모리를 단기 메모리와 함께 사용하여 행동을 예측합니다.

Long- and Short-Term Memory 관리 방식으로는, 단기 메모리는 최근 관찰된 프레임들을 소량 저장하는 반면, 장기 메모리는 더 오래된 프레임들을 저장하여 인코더의 입력으로 사용됩니다. 예를 들어, 장기 메모리 $m_L$는 2048개의 슬롯을 가지며 이는 4 FPS의 샘플링 속도로 약 512초에 해당하는 비디오 콘텐츠를 저장하고, 단기 메모리 $m_S$는 32개의 슬롯을 가지며 약 8초에 해당합니다.

이 방식은 트랜스포머(Transformers)의 유연성을 활용하여 행동 이해를 위한 장기 및 단기 정보를 효율적으로 결합하는 여러 문제를 해결합니다.

### 3.2 Long- and Short-Term Memories




### 3.3 LSTR Encoder

LSTR encoder의 역할은 $m_L$을 decoding 할 수 있는 latent representation으로 encoding 하는 과정. ($m_L \times C \to n_1 \times C$)

- attention with linear complexity
- two-stage memory compression

**The Transformer decoder unit**
When,
- C is embedding space,
- $\lambda \in \mathbb{R}^{n \times C}$,
- $\theta \in \mathbb{R}^{m \times C}$.

$$\lambda' = SelfAttn(\lambda) = Softmax (\frac{\lambda \cdot \lambda^T}{\sqrt{C}})\;\lambda$$
$$CrossAttn(\sigma(\lambda'), \theta) = Softmax (\frac{\sigma(\lambda') \cdot \theta^T}{\sqrt{C}}) \; \theta $$
( $\sigma(\lambda')$ is query of $\lambda'$ )

두 식을 합한 unit의 시간복잡도는 $O(n^2C + nmC)$이다. 여기서 $n \ll m$ 이 성립하게 되면, m에 linear한 complexity를 가지게 된다.


**Two-stage memory compression**
1. One Transformer decoder unit with $n_0$ output tokens
2. $\ell_{enc}$ stacked Transformer decoder units with $n_1$ output tokens

($1 + \ell_{enc}$)-layer의 transformer는 $O(m_L^2(1+\ell_{enc})C)$의 시간 복잡도를 가지고, 
n개의 output token을 가지는 transformer의 경우 $O((n^2+nm_L)(1+\ell_{enc})C)$가 된다.

제안하는 LSTR 인코더는 $O(n_0^2C+n_0m_LC+(n_1^2+n_1n_0)\ell_{enc}C)$의 시간복잡도를 가지게 되는데, $n_0, n_1 \ll m_L$이고, $\ell_{enc}$의 경우 1보다 항상 크기 때문에, two-stage memory compression이 효율적이다.

### 3.3 LSTR Decoder

LSTR decoder는 short-term 메모리를 query로 사용해서 '인코딩된 long-term 메모리'에서 유용한 정보를 검색하는 과정.

$\ell_{dec}$개의 transformer를 쌓아서 생성.
LSTR Encoder의 output을 입력으로, $m_S$의 feature vector를 query 입력으로,
$m_S$개의 probability vector를 출력으로 한다. $\{p_T, \cdots , p_{T-m_s+1}\} \in [0,1]^{K+1}$
K개의 action category와 "background(=nothing)" class를 classification할 수 있다.

### 3.5 Online Inference with LSTR

매 비디오 프레임미다 LSTM encoder를 실행하면 높은 시간 복잡도를 가진다.
첫번째 memory compression에서 $O(n_0^2C + n_0m_LC)$,
두번째 memory compression에서 $O(n_1^2+n_1n_0\ell_{enc}C)$.

따라서, 첫번째 memory compression을 미리 계산하는 방식으로 효율성을 높인다.
첫번째 transformer unit의 query는 고정이므로, self-attention 출력을 한번만 계산하고 추론 시에 계속 재사용할 수 있다.

$$CrossAttn(q_i, \{f_{T-\tau} + s_{\tau}\}) = \sum_{\tau=m_S}^{m_S + m_L - 1} \frac{\exp(({f_{T-\tau} + s_{\tau}) \cdot q_i}/{\sqrt C})} {\sum_{\tau=m_S}^{m_S + m_L - 1} \exp{((f_{T-\tau} + s_{\tau}) \cdot q_i}/{\sqrt C})} \cdot (f_{T-\tau} + s_{\tau})$$
$f$는 video frame feature, $s$는 위치 임베딩
위치 임베딩은 추론 중 변화하지 않으므로, 미리 계산이 가능하다.

As를 미리 계산하고 큐를 유지함으로써 주의 가중치 계산이 O(n₀(mL + C))로 줄어들어, 기존의 O(n₀ × mL × C)보다 효율적입니다.
    - 비록 가중합 때문에 크로스 어텐션의 복잡도는 여전히 O(n₀mLC)이지만, 특히 C(특징 차원 수)가 1024 이상인 경우가 많아 성능 개선은 상당합니다.
### Limitations
However, we note that LSTR is operating only on the temporal dimension.
An end-to-end video understanding system requires simultaneous spatial and temporal modeling for optimal results. 
Therefore extending the idea of LSTR to spatio-temporal modeling remains an open yet challenging problem.