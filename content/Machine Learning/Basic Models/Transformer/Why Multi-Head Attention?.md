---
title: Why Multi-Head Attention?
---
Single self-attention은 한개의 세트에 대한 1개의 $W_q, W_k, W_v$ 가 존재한다.

이와 같은 구조의 단점은 1가지 방법만으로 similarity를 측정한다는 점이다.

1장의 image에 대해서, color, texture, shape 등의 similarity를 측정하기 위해서, 여러 개의 head를 사용하고자 하는 것이 multi-head attention의 방법이다.

따라서 각각을 num-heads 갯수만큼으로 늘리게 되고, $W_q^1, W_q^2, … ,W_q^h$ , $W_k^1, W_k^2, … ,W_k^h$, $W_v^1, W_v^2, … ,W_v^h$ 로 사용한다.

마지막에 $W^O$ 를 사용하여 모든 head의 내용을 merge한다.

$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O \\
\\
\text{where } head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
$$

본문에도 나와있지만 한번 더 그림으로 표현해보자면,

![[Transformer_7.png]]

각 검은색 화살표가 1개의 head 입력을 의미한다.

파란색으로 나눠지는 것은 기존의 (196, 768) embedding을 (196, 12, 64)로 나눠서 multi-head attention에 넣게 된다는 표현이다. \
(196 patch / 764 dim vector -> 196 patch / 12 head, 64 dim vector) 

각각의 head들에는 이미지의 특정한 attribute 정보가 들어 있을 것으로 예상된다.

[Interpreting CLIP's Image Representation via Text-Based Decomposition](https://arxiv.org/abs/2310.05916) 논문에서는 ViT 구조로 만들어진 CLIP이라는 모델에 대해서 각각의 head가 어떤 정보를 가지고 있는지에 대해서 분석한다.

이 논문에 대해서는 나도 정리한 바 있다. \
[rhqo.github.io - Interpreting-CLIP's-Image-Representation-via-Text-Based-Decomposition](https://rhqo.github.io/Computer-Vision/Interpreting-CLIP/Text-Based-Decomposition/Interpreting-CLIP's-Image-Representation-via-Text-Based-Decomposition)
