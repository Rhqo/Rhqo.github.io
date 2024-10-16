# TSM: Temporal Shift Module for Efficient Video Understanding

TSM은 데이터 이동과 연산을 분리하여 효율성을 높이는 모듈입니다. 일반적인 컨볼루션 연산에서는 단순한 시프트(shift) 동작이 효율적이지 않습니다. 이를 해결하기 위해 우리는 데이터 이동을 최소화하고 모델 용량을 증가시키는 두 가지 기술을 제안하여 효율적인 TSM 모듈을 설계했습니다.

![[TSM_0.png]]
#### 3.1 Intuition
TSM의 직관적인 개념을 설명하기 위해, 먼저 일반적인 1D 컨볼루션을 예로 듭니다. 커널 크기가 3인 1D 컨볼루션의 가중치 W=(w1,w2,w3)라고 가정하고, 입력 X는 무한 길이의 1D 벡터라고 할 때, 컨볼루션 연산 Y=Conv(W,X)은 다음과 같이 표현할 수 있습니다:

$$Yi=w1⋅Xi−1+w2⋅Xi+w3⋅Xi+1​
$$
이 컨볼루션 연산은 두 단계로 분리될 수 있습니다: **shift** 와 **multiply-accumulate**. 입력 X를 각각 −1, 0, +1 만큼 시프트하고, 가중치 w1,w2,w3를 곱한 후 합산하면 Y가 됩니다. 수식으로 표현하면 다음과 같습니다:

- shift 연산:
	$$X_i^{−1}=X_{i−1},X_i^0=X_i,X_i^{+1}=X_i^{+1}​$$ 
- multiply-accumulate 연산:
    $$Y=w_1⋅X^{−1} + w_2⋅X^0 + w_3⋅X^{+1}$$

첫 번째 단계인 shift는 곱셈 없이 수행할 수 있습니다. 두 번째 단계인 multiply-accumulate은 더 많은 계산 비용이 들지만, TSM은 이 multiply-accumulate 연산을 뒤따르는 2D 컨볼루션에 병합하여, 2D CNN 기반 모델과 비교했을 때 추가 비용을 유발하지 않습니다.

오프라인 비디오 인식은 시간적 차원에서 일부 채널을 −1 방향으로, 일부는 +1 방향으로 shift하고 나머지는 그대로 두는 방식으로 작동합니다.
온라인 비디오 인식의 경우, 미래 프레임에 접근할 수 없기 때문에 과거 프레임에서 미래 프레임으로만 단방향 shift를 수행합니다.
#### 3.2 Naive Shift Does Not Work
Naive shift 전략은 두 가지 문제를 발생시킵니다. 
- 첫째, 데이터 이동이 많아져 메모리 사용량과 지연 시간이 증가합니다. 
- 둘째, 채널의 정보를 다른 프레임으로 이동시켜 공간적 특성 학습 능력이 감소합니다. 
이러한 이유로 naive shift는 성능과 효율성을 모두 저하시킵니다.

#### 3.3 모듈 설계
이 문제를 해결하기 위해 두 가지 기술을 제안합니다:

1. **Reducing Data Movement**: 모든 채널을 시프트하는 대신 **일부** 채널만 시프트하여 메모리 이동 비용을 줄입니다. 예를 들어, 1/8만 시프트할 경우 지연 시간 오버헤드는 3%로 줄어듭니다.
    
2. **Keeping Spatial Feature Learning Capacity**: 모듈을 'Residual block' 안에 넣는 **'residual shift'** 전략을 사용하여 공간적 특성 학습 능력을 유지합니다. 실험 결과, 잔여 시프트는 모든 비율의 시프트에서 더 나은 성능을 보여주며, 특히 채널의 1/4을 시프트할 때 성능이 최적화됩니다.