---
title: 
tags:
---
# Text-Only Training for Image Captioning using Noise-Injected CLIP
> [!Abstract] Abstract
> We consider the task of image-captioning using only the CLIP model and additional text data at training time, and no additional captioned images. Our approach relies on the fact that CLIP is trained to make visual and textual embeddings similar. Therefore, we only need to learn how to translate CLIP textual embeddings back into text, and we can learn how to do this by learning a decoder for the frozen CLIP text encoder using only text. We argue that this intuition is “almost correct” because of a gap between the embedding paces, and propose to rectify this via noise injection during training. We demonstrate the effectiveness of our approach by showing SOTA zero-shot image captioning across four benchmarks, including style transfer.
> 
> 우리는 학습 시 CLIP 모델과 추가 텍스트 데이터만을 사용하여 이미지 캡션 작업을 고려하며, 추가 캡션된 이미지는 고려하지 않습니다. 우리의 접근 방식은 CLIP이 시각적 임베딩과 텍스트 임베딩을 유사하게 만들도록 훈련된다는 사실에 의존합니다. 따라서 CLIP 텍스트 임베딩을 텍스트로 다시 번역하는 방법만 배우면 되며, 이를 수행하는 방법은 텍스트만을 사용하여 고정된 CLIP 텍스트 인코더의 디코더를 학습함으로써 배울 수 있습니다. 우리는 임베딩 속도 사이의 간격 때문에 이러한 직관이 "거의 정확하다"고 주장하며, 학습 중 노이즈 주입을 통해 이를 수정할 것을 제안합니다. 우리는 스타일 전이를 포함한 네 가지 벤치마크에서 SOTA 제로샷 이미지 캡션을 보여줌으로써 우리의 접근 방식의 효과를 입증합니다.


