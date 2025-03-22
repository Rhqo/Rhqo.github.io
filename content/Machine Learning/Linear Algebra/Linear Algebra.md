---
title: Linear Algebra
tags:
  - section
---
# Preface

[MIT 18.06 Linear Algebra Youtube](https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8)

두 column 벡터 $a_1, a_2$ 와 $\text{zero vector}$를 가정하고 시작하자. \
이들은 3차원 공간의 한 점에 대응한다.

이 두 벡터를  $\text{Linear Combination}$을 사용하여 나타낼 수 있으며, 이 combination들은 한 평면을 모두 차지한다. \
3차원 공간상의 무한한 평면이다.
$$
	\text{Linear Combination} = ca_1 + da_2 \; \text{ for any numbers c and d}
$$
![[preface_0.png]]

이제 linear algebra의 fundamental idea가 등장한다 : **a matrix**
Matrix $A$ 는 n개의 column vectors $a_1, a_2, a_3, ..., a_n$ 을 가지며, 위의 경우 3차원 공간상의 두 column vector $a_1, a_2$ 를 의미한다. \
따라서, $A$는 3개의 row($dimension$)와 2개의 column($n$)을 가진다. \
3차원 공간상의 2개의 column의 combination들은 plane을 생성하며, 이를 matrix의 ==column space==라고 부른다.

모든 $A$에 대해서, $A$의 column space는 
