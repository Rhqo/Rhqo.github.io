# Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams

```embed
title: "Flash-VStream: Memory-Based Real-Time Understanding for Long Video Streams"
image: "https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-fb.png"
description: "Benefiting from the advancements in large language models and cross-modal alignment, existing multi-modal video understanding methods have achieved prominent performance in offline scenario. However, online video streams, as one of the most common media forms in the real world, have seldom received attention. Compared to offline videos, the ‘dynamic’ nature of online video streams poses challenges for the direct application of existing models and introduces new problems, such as the storage of extremely long-term information, interaction between continuous visual content and ‘asynchronous’ user questions. Therefore, in this paper we present Flash-VStream, a video-language model that simulates the memory mechanism of human. Our model is able to process extremely long video streams in real-time and respond to user queries simultaneously. Compared to existing models, Flash-VStream achieves significant reductions in inference latency and VRAM consumption, which is intimately related to performing understanding of online streaming video. In addition, given that existing video understanding benchmarks predominantly concentrate on offline scenario, we propose VStream-QA, a novel question answering benchmark specifically designed for online video streaming understanding. Comparisons with popular existing methods on the proposed benchmark demonstrate the superiority of our method for such challenging setting. To verify the generalizability of our approach, we further evaluate it on existing video understanding benchmarks and achieves state-of-the-art performance in offline scenarios as well. All code, models, and datasets are available at the https://invinciblewyq.github.io/vstream-page/"
url: "https://arxiv.org/abs/2406.08085v1"
```

![[Flash-VStream_0.png]]
![[Flash-VStream_1.png]]
# 3. Flash-VStream
## 3.1 Streaming visual encoder
연속적으로 visual information을 embedding feature들로 인코딩
CLIP ViT-L 사용
Patch token들은 훈련과 추론 과정에서만 사용
Frame Stream $\{V^t\}^{\infty}_{t=1}$ 에서, t번째 프레임의 encoder map은 $V^t \in \mathbb{R}^{H\times W \times 3}$ to feature map $e^t \in \mathbb R^{ P \times P \times D}$ P는 ViT의 patch token의 수이며, D는 ViT의 dimension이다.

## 3.2 **S**patial-**T**emporal-**A**bstract-**R**etrieved memory
### Spatial memory $M_{spa} \in \mathbb{R}^{N_{spa} \times P^2_{spa} \times D}$
가장 최근의 detailed spatial 정보들을 가짐
short-term에 사용
FIFO queue로 구현
$$M^t_{spa} = M^t_{buff}[0:N_{spa},:,:] \tag{2}$$
세분화된 spatial data에 즉시 접근 가능
### Temporal memory$M_{tem} \in \mathbb{R}^{N_{tem} \times P^2_{tem} \times D}$
시간에 따른 dynamic 정보들을 통합함
long-term에 중요
사이즈가 $N_{tem}$을 능가하게 되면, $g_{wkmeans}$(Weighted K-means Clustering)알고리즘 적용
$$M^t_{tem} = g_{wkmeans}(concat(g_{pooling}(e^t,P_{tem}),M^{t-1}_{tem}),N_{tem})\tag{3}$$
-> 이 적용은 메모리 내용을 $N_{tem}$ 클러스터로 응축하는데, 이는 비디오의 주요 사건에 대한 표현으로 볼 수 있다. 
$N_{tem}$ 클러스터의 센트로이드가 시간 맥락을 효율적으로 저장하기 위한 새로운 메모리로 사용된다.

### Abstract Memory $M_{abs} \in \mathbb{R}^{N_{abs} \times P^2_{abs} \times D}$
Semantic Attention model ($f_{SA}$)를 사용하여 high-level의 semantic concept 해석을 돕는다.
spatial, temporal memory로부터의 통찰력을 추상화되고 실행가능한 지식으로 종합한다.
$f_{SA}$는 가장 최근의 feature를 사용해 전체 비디오의 시놉시스인 $M_{abs}$를 조정한다.
$$M^t_{abs} = f_{SA}(M^{t-1}_{abs}, g_{pooling}(e^t, P_{abs}), N_{abs})\tag{4}$$


### Retrieved memory $M_{ret} \in \mathbb{R}^{N_{ret} \times P^2_{spa} \times D}$
P에 spa는 오타 아님.
가장 중요한 프레임 특징을 식별하고 검색하여 정확한 spatial 세부 정보를 회상하는 데 집중한다.
먼저 시간 메모리 $M_{tem}$에서 N개의 클러스터 중 상위 K(K는 $N_{ret}$와 같음) 큰 클러스터를 선택합니다. 그런 다음 이 K 클러스터의 중심에 가장 가까운 특징 버퍼의 프레임 특징을 검색하여 시간 메모리를 더욱 상세한 공간 정보로 보충한다.
$$
M_{t}^{ret} = g_{retrieve}(M^t_{buff}, \mathbf{M}^{t}_{tem}, N_{ret})\tag{5}
$$


new feature $e^t$는 STAR메모리로 다음과 같이 이동한다.
$$
M^t_{buff} = concat(g_{pooling}(e^t, P_{spa}), M^{t-1}_{buff})[0 : N_{buff}, :, :] \tag{1}
$$
$$(1) \rightarrow (2) \rightarrow (3) \rightarrow (4) \rightarrow (5)$$

최대 메모리 사이즈는 다음과 같을 것.
$$\text{MAXSIZE} = (N_{spa}+N_{ret}) \times P^2_{spa} + N_{tem} \times P^2_{tem} + N_{abs} \times P^2_{abs}$$
## 3.3 Real-time LLM decoder
t 시간의 질문 $Q^t$에 대해서, LLM decoder는 text embedding $I^t_{text}=f_{embed}(Q^t)$를 계산하고, 
STAR 메모리에 $M^t$에 대해 projector를 사용하여 vision embedding $I^t_{vision}=f_{proj}(M^t)$를 계산한다.
이후 real-time의 답변을 생성한다. ($A^t = f_{LLM}(I^t_{text}, I^t_{vision}).decode()$)