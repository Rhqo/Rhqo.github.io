왜 이러한 현상이 발생하는가? \
3차원에서 2차원으로 projection시키는, 비정사각행렬인 W를 가정해보자.

Linear model에 대해서, $W^TW$ 가 $I$ 에 수렴하게 될 것이므로, $W$ 의 열벡터가 linearly independent 해질 것이다.
![[Why-does-it-happened_3.png]]


$2\times3$ 행렬인 W에 대해서, \
$W$의 column vector 2개는 3차원에서 표현한 2차원의 basis가 될 것이고, \
$W$의 row vector 3개는 2차원에서 표현한 3차원 basis가 될 것이다.
![[Why does it happended?_0.png]]

Linear model의 projection, hidden layer를 시각화해보면, 다음과 같다.
![[Why does it happened?_1.png]]
