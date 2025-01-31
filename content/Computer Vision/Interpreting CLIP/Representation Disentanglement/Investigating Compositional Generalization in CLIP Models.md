
> [!Abstract] Abstract
> CLIP models have recently shown to exhibit Out of Distribution (OoD) generalization capabilities. However, Compositional Out of Distribution (C-OoD) generalization, which is a crucial aspect of a model’s ability to understand unseen compositions of known concepts, is relatively unexplored for the CLIP models. Our goal is to address this problem and identify the factors that contribute to the C-OoD in CLIPs. We noted that previous studies regarding compositional understanding of CLIPs frequently fail to ensure that test samples are genuinely novel relative to the CLIP training data. To this end, we carefully synthesized a large and diverse dataset in the single object setting, comprising attributes for objects that are highly unlikely to be encountered in the combined training datasets of various CLIP models. This dataset enables an authentic evaluation of C-OoD generalization. Our observations reveal varying levels of C-OoD generalization across different CLIP models. We propose that the disentanglement of CLIP representations serves as a critical indicator in this context. By utilizing our synthesized datasets and other existing datasets, we assess various disentanglement metrics of text and image representations. Our study reveals that the disentanglement of image and text representations, particularly with respect to their compositional elements, plays a crucial role in improving the generalization of CLIP models in out-of-distribution settings. This finding suggests promising opportunities for advancing out-of-distribution generalization in CLIPs.
> 
> CLIP 모델은 최근 분포 외 일반화(Out of Distribution, OoD) 능력을 보이는 것으로 나타났습니다. 그러나 조합 분포 외(Compositional Out of Distribution, C-OoD) 일반화는 CLIP 모델이 알고 있는 개념의 보지 못한 조합을 이해하는 모델의 능력에서 중요한 측면이지만, 상대적으로 탐구되지 않았습니다. 우리의 목표는 이 문제를 해결하고 CLIP에서 C-OoD에 기여하는 요소를 식별하는 것입니다. 우리는 CLIP의 조합 이해에 대한 이전 연구들이 테스트 샘플이 CLIP 훈련 데이터에 비추어 보아 진정으로 새로운 것인지 확인하는 데 자주 실패했다는 점을 주목했습니다. 이를 위해 우리는 단일 객체 설정에서 다양한 속성을 포함한 대규모의 다양한 데이터셋을 신중하게 합성했습니다. 이 데이터셋은 다양한 CLIP 모델의 결합 훈련 데이터셋에서 만날 가능성이 매우 낮은 객체들의 속성을 포함하고 있습니다. 이 데이터셋은 C-OoD 일반화에 대한 진정한 평가를 가능하게 합니다. 우리의 관찰은 다양한 CLIP 모델에서 C-OoD 일반화의 수준이 다르다는 것을 드러냅니다. 우리는 CLIP 표현의 분리 해제가 이 맥락에서 중요한 지표로 작용한다고 제안합니다. 우리의 합성 데이터셋과 기존 데이터셋을 활용하여 텍스트와 이미지 표현의 다양한 분리 해제 메트릭스를 평가합니다. 우리의 연구는 이미지와 텍스트 표현의 분리 해제, 특히 그들의 조합 요소와 관련하여 CLIP 모델의 분포 외 설정에서 일반화를 향상시키는 데 중요한 역할을 한다는 것을 보여줍니다. 이 발견은 CLIP의 배포 외 일반화를 진전시킬 수 있는 유망한 기회를 제안합니다.

# Introduction

보통의 앰뷸런스 사진은 빨간색의 특징을 가지고 있다. (In-Distribution)

하지만, 앰뷸런스 사진 중 빨간색이 아닌 특징을 가지고 있는 것들도 있다. (Out-of-Distribution)

⇒ 빨간색이 아닌 앰뷸런스에 대해서도 찾을 수 있게 되어, 앰뷸런스에 대한 robust한 성능을 가지게 하는 것이 OoD Generalization이 될 것이다.

CLIP은 OoD 성능이 좋다.

본 논문에서는, **Compositional Out-of-Distribution (C-OoD) Generalization**을 시도하고자 한다.

기존의 알고 있는 combination으로 부터, unseen combination의 객체까지 일반화하는 것을 의미한다.

ex) Red + Ambulance → Decorated + Ambulance

이 논문이 질문하는 것은 크게 2가지 이다.

- CLIP이 single object setting에서 nontrivial(비자명)한 CoOD generalization 능력을 가지고 있는지?
- CLIP 모델에서 이 능력은 어디에서 비롯된 것인지?

따라서, 본 논문에서는 다음을 제안한다.

- 일반적인 CLIP 학습 데이터셋에서는 볼 수 없는 attribute-object 쌍의 이미지 테스트 데이터셋 설계
- 신중하게 설계되고 제어된 환경에서 다양한 CLIP의 compositional generalization을 benchmarking
- 벤치마크에서 더 나은 성과에 기여하는 요인 분석

# Methodology

## ImageNet-AO dataset

![[COoD_0.png.png]]

기존 dataset에서는 보이지 않는 조합들로 **생성된** 이미지들(synthetic data)의 dataset이다.

다음은 이 dataset의 생성 과정이다.

![[COoD_1.png.png]]

### Generation Phase
1. Selection of Objects (Nouns)
    ImageNet dataset으로부터 class name을 가져온다. (Object)
2. Selection of Attributes (Adjectives)
    Visual Attributes Words(VAW) 데이터셋으로부터 아래와 같은 140개의 형용사를 가져온다. (Attribute)
    > [!Tag] Attributes
    > cracked, dilapidated, dry, folded, wet, jagged, moss covered, rough, textured,wrinkled, transparent, clean, dirty, dusty, stained blue plaid, checkered, dotted,floral, lined, red striped, speckled, spotted, striped, arch shaped, arrow shaped,circular, conical, cubed, curved, curly, cylindrical, diamond shaped, domed, heart shaped, octagonal, oval shaped, rectangular, round, rounded, spherical, spiky, spiral, square, triangular, aluminum, asphalt, bamboo, brass, brick, cardboard, cement, ceramic, chocolate, chrome, clay, cloth, cobblestone, concrete, denim, dirt, fabric, fluffy, foamy, furry, glass, granite, gravel, hardwood, iron, jean, khaki,leather, marble, metal, muddy, paper, pebbled, plastic, plush, porcelain, red brick, rocky, rubber, sandy, silk, snowy, stainless steel, steel, stone, straw, stucco, styrofoam, tiled, wicker, wooden, water, colorful, red, pink, purple, green, amber, aqua, beige, black, blond, blue, bluish, bronze, brown, burgundy, fuchsia, golden, gray, green, ivory, maroon, murky, orange, pink, purple, purplish, red, reddish, silver, tan, taupe, teal, terracotta, turquoise, violet, white, yellow

3. Image Generation with Attribute-Object Prompts
    140개의 형용사와, 1000개의 명사를 조합하여, 140,000의 unique한 pair를 만들어 prompt를 생성한다. \
    SD-XL Turbo 모델을 사용하여 420,000개의 이미지를 생성한다.

### Filtering Phase
1. Initial Validation
    사람이 판단하여 filtering
    
2. Exclusion of Known Combinations
    LAION, CommonPool (DataComp), YFCC15m, CC12m 등의 dataset에서 비슷한 이미지가 있는 경우 제거
    
3. Verification of OoD Status
    기존의 dataset과 생성한 dataset의 K-nearest neighbors search를 하여 matching이 되지 않는다면, 생성한 이미지들이 충분히 unique함을 알 수 있으므로, 저장한다.
    
최종적으로 60,000장의 이미지 생성

![[COoD_2.png.png]]

충분히 다른 데이터셋과 비교해서 차별화되었음을 볼 수 있으며,

OoD dataset이라고 볼 수 있겠다.

# Comparison of CLIP Models on ImageNet-AO

이미지와 텍스트 임베딩의 cosine similarity를 고려하여 성능 평가

1. attribute + object 를 다양한 template을 통해 class당 80개의 caption을 생성
2. caption들의 평균 계산하여 final embedding 구한다.
3. final embedding을 cosine similarity에 사용

![[COoD_3.png.png]]
위 이미지는 성능 평가 결과이며,
기존 dataset (ImageNet)에서 성능이 올라갈수록, OoD의 성능도 증가하는 모습이다.

# Why CLIP has Compositional Generalization?

기존의 COoD generalization은 disentangled representation일 수록 잘 된다는 기존 연구가 있다.

이러한 사실을 바탕으로, language의 분리되는 특성, 다양하고 큰 training dataset을 통해

→ text representation이 decomposable하고, 이것이 contrastive learning을 통해 align되면서, 이미지단으로 전파된다.

⇒ 즉, CLIP이 Compositional generalization할 수 있는 능력의 원인은,

- **CLIP 텍스트 임베딩의 decomposabilty는 CLIP C-OoD 일반화와 상관관계가 있다.**
- **텍스트와 이미지 표현의 상호 정보를 contrastive learning을 통해 암묵적으로 최대화함으로써 이미지 인코딩에서 텍스트 representation의 disentanglement가 유도된다.**

이를 실험과 수식을 통해 증명했다.

### 실험
![[COoD_4.png.png]]
Represent의 disentanglement를 측정하는 Z-diff score를 보면, text represent의 z diff score가 매우 높았다.

또한, epoch가 진행되면서 image encoder의 z-diff score가 증가하는 것을 보면, text의 disentanglement가 이미지로 전파된다고 생각할 수 있을 것이다.

### 수식
- $y_1$ 과 $y_2$ 는 각각 object와 attribute의 text embedding
- $x_1$ 과 $x_2$ 는 대응하는 image embedding
- decomposable text embedding의 의미를 $y_1 \perp y_2$ 라고 가정
- Contrastive loss를 minimize하면, mutual information $I(x_1, x_2; y_1, y_2)$ 는 maximize될 것
![[COoD_5.png.png]]
- $y_1 \perp y_2$ 이기 때문에 결국 $x_1$ 과 $x_2$ 는 independent 해질 것이다.
- 따라서, 만약 $y$ 가 이미 decomposed되어 있다면, $I(x_1, x_2; y_1, y_2)$ 를 최대화하는 것은 $x$ 를 decomposing하는 것 과 같다.

# Decomposable representation of CLIP Models

## 실험 1. Attribute-Object Decomposition of Representation Space

### Disentanglement of Attributes and Objects
![[COoD_6.png.png]]
CLIP의 Image와 text의 embedding이 representation disentanglement와 상관관계가 있다.

COoD generalization 성능이 올라갈수록, disentanglement metric이 증가한다.

→ imagenet-ao에서 성능이 높을수록 disentangle 하다.

### Intrinsic Dimensionality of the Composition Representations
학습 없이, image embedding을 평가하기 위해, image embedding을 여러개 쌓아서 matrix를 만든 다음 soft rank 한다.

Soft rank는 embedding space의 relative intrinsic dimensionality를 나타낼 수 있다.

Embedding이 disentangled 일 경우 값이 낮게 나올 것이다.

Entangled 되었을 경우 full-rank에 가까워져 값이 높아질 것이다.

![[COoD_7.png.png]]
실험 결과는 위과 같고, COoD generalization 성능이 높아지면, soft rank도 낮아져,
**text와 image 둘다 disentangled 되었다**고 볼 수 있다.

## 실험 2. Attribute-Object Decomposition of Representation Space

### Image retrieval with image + text queries
![[COoD_8.png.png]]

## 실험 3. Disentanglement of Fine-Grained Factors

### For in-depth analysis of the fine-grained disentanglement
![[COoD_9.png.png]]
Fine-Grained 요소를 가진 것들도 disentangle 한지 판단하기 위해, 위의 2 dataset에 대해서 평가했다.

결과는 아래와 같고, fine-grained 요소들에 대해서도 disentangle하다.
![[COoD_10.png.png]]
### More Analysis on decomposability of the representation space
![[COoD_11.png.png]]
