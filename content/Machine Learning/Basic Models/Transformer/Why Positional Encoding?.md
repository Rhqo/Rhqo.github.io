---
title: Why Positional Encoding?
---
Position vector를 부여하는 방법으로 다음과 간단한 두 가지 방법을 떠올려 볼 수 있다.

1. 첫번째 token에는 1, 두번째 token에는 2, ... 방식으로 sequence 크기에 비례해서 일정하게 커지는 정수 값 부여한다.
2. 첫번째 token에는 0, 마지막 token에는 1을 부여하여, 그 사이를 (1/시퀀스 길이)로 나누어 나온 값을 부여한다.

1번의 경우, position vector가 무한히 커질 수 있게 된다. \
Position vector가 커지면, patch embedding과 더했을 때, patch의 정보보다 position 정보가 지배적이게 되어 patch의 정보가 무시될 수 있다. \
또한, position vector가 일정한 범위를 갖게 되지 않아서 모델의 generalize 역시 힘들어진다.

2번의 경우, 같은 위치의 token이 같은 position vector를 가지지 않게 된다. \
시퀀스의 길이가 3인 경우에는 2번째 position vector가 0.5가 되지만, \
시퀀스의 길이가 4인 경우에는 2번째 position vector가 0.3이 된다. \
또한, 시퀀스의 총 길이도 알기 힘들고, 바로 옆에 위치한 token들 간의 position vector의 차이 역시 달라지게 된다.

이러한 점을 고려하여, position vector는 다음 2가지 조건을 모두 만족하여야 한다.
- **Token의 기존 정보가 무시되지 않도록 position vector값이 너무 크면 안된다.**
- **같은 위치의 token은 항상 같은 position vector값을 가지고 있어야 한다.**

이 두가지 조건을 만족하는 최적의 함수는 **-1~1 값을 반복**하고, **주기함수**라는 특징을 가진 **sinusoidal 함수**가 된다.

-> 여기서 -1~1 값을 가지는 sigmoidal 함수가 되지 않는 원인에 대해서 생각해 볼 수 있다. \
Sigmoidal 함수는 시퀀스의 길이가 길어질 경우, position vector의 차가 거의 없어지는 문제가 발생할 수 있기 때문에 사용할 수 없다.

Sinusoidal 함수는 -1~1을 반복하는 주기함수 이기 때문에, 서로 다른 위치에 있는 두 token이 같은 position vector를 가지게 되는 경우가 생기게 된다.

Position vector가 1d의 정보가 아니라는 점을 고려하여, 이를 해결할 수 있는데, 그 방법은 바로 다양한 주기의 sinusoidal 함수를 동시에 사용하는 방법이다.

$$
p_x = [\sin(x), \sin(x/B), \sin(x/B^2), ..., \sin(x/B^p)]^T \\
p_y = [\sin(y), \sin(y/B), \sin(y/B^2), ..., \sin(y/B^p)]^T \\
p = \begin{bmatrix}p_x \\ p_y\end{bmatrix}
$$
![[Why Positional Encoding?_0.png]]

마지막 한가지 더 다른 방법은 model에 의해 positional code를 학습시키는 것이다. \
Space를 represent 하는 데 sinusoidal code보다 유리할 수 있다.

# Why add position encoding?

위의 내용까지는 position vector를 어떻게 설정할 것인지에 대해서 생각해 본 내용이다.

하지만 또 다른 의문이 남아 있다.

Position encoding과 patch embedding을 사용하여 token embedding을 할 때, 왜 concatenation 연산이 아닌, summation 연산을 사용한 걸까?

이에 관련하여 아직 명확히 정리된 것은 없다.

하지만 summation 연산을 수행함으로써 메모리 공간 상의 이득이 있고, min frequency가 1/10000인 상태에서 시퀀스 길이가 짧을 경우 position vector가 sparse 해지기 때문에, summation을 수행하더라도 patch embedding의 정보가 손실되지 않을 것이라고 추정된다.