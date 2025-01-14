왜 이러한 현상이 발생하는가? \
3차원에서 2차원으로 projection시키는, 비정사각행렬인 W를 가정해보자.

Linear model에 대해서, $W^TW$ 가 $I$ 에 수렴하게 될 것이므로, $W$ 의 열벡터가 linearly independent 해질 것이다.
![[Why-does-it-happened_3.png]]

하지만, ReLU output model에 대해서, $W^TW$ 는 $I$ 에 수렴하게 될 필요성이 떨어지게 되므로, $W$ 의 열벡터가 linearly independent 해질 이유가 없어진다. \
결과적으로 $W$ 는 어느정도의 자유도를 갖게 될 것이다.
![[Why-does-it-happened_4.png]]

$2\times3$ 행렬인 W에 대해서, \
$W$의 column vector 2개는 3차원에서 표현한 2차원의 basis가 될 것이고, \
$W$의 row vector 3개는 2차원에서 표현한 3차원 basis가 될 것이다.
![[Why does it happen?_0.png]]

Linear model의 $W$를 시각화해보면, 다음과 같다.
![[Why does it happen?_1.png]]
평면은 $x_1$과 $x_2$를 span한 평면으로, projection되는 평면을 의미한다. ($x_1$과 $x_2$가 basis) \
$x’_1$은 $x’_2$와 서로 orthogonal 하며, $x’_3$는 거의 활성화되지 않는 모습을 보인다.

다음으로 ReLU output model의 $W$를 시각화해보면, 다음과 같다.
![[Why-does-it-happened_2.png]]
평면은 $x_1$과 $x_2$를 span한 평면으로, projection되는 평면을 의미한다. ($x_1$과 $x_2$가 basis) \
$x’_1$은 $x’_2$, $x’_3$와 모두 orthogonal 하며, $x’_2$와 $x’_3$는 서로 antipodal pair가 되는 모습을 보인다.

그렇다면, projection 평면을 위와 같이 조절했을 때, 구체적으로 loss가 어떤 형식으로 생성되는지를 생각해보자. \
위의 예시의 경우, linear model은 x, y축에 대한 loss는 거의 없을 것이고, z축에 대한 loss는 클 것으로 예상되기 때문에, xy 좌표는 고정하고 z좌표가 변화할 때의 loss를 예시로 들어보겠다.

![[Why does it happen?_5.png]]

ReLU output model은 Linear model에 비해 z축에 대한 loss가 천천히 증가하는 모습이다.

$W^TWx+b$ 의 Linear model과 $ReLU(W^TWx+b)$ 의 ReLU output model을 forward를 사용하여, 기존의 좌표를 얼마나 잘 복구하는지를 살펴보면, 다음 두 그림과 같은 결과가 나오게 된다.
