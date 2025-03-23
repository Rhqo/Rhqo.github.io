---
title: Linear Algebra
tags:
  - section
---
# Index
- ### 1. Vectors and Matrices
	- 1.1 Vectors and Linear Combinations
	- 1.2 Lengths and Angles from Dot Products
	- 1.3 Matrices and Their Column Spaces
	- 1.4 Matrix Multiplication $AB$ and $CR$
- ### 2. Solving Linear Equations $Ax=b$
	- 2.1 Elimination and Back Substitution
	- 2.2 Elimination Matrices and Inverse Matrices
	- 2.3 Matrix Computationsand $A=LU$
	- 2.4 Permutations and Transposes
	- 2.5 Derivatives and Finite Difference Matrices
- ### 3. The Four Fundamental Subspaces
	- 3.1 Vector Spacesand Subspaces
	- 3.2 Computing the Nullspace by Eliminationn: $A=CR$
	- 3.3 The Complete Solutionto $Ax =b$
	- 3.4 Dimensions of the Four Subspaces
	- 3.5 Dimensions of the Four Subspaces
- ### 4. Orthogonality
	- 4.1 Orthogonality of Vectors and Subspaces
	- 4.2 Projections onto Lines and Subspaces
	- 4.3 Least Squares Approximations
	- 4.4 Orthonormal Bases and Gram-Schmude
	- 4.5 The Pseudoinverse of a Matrix
- ### 5. Determinants
	- 3 by 3 Determinants and Cofactors
	- Computing and Using Determinants
	- Areas and Volumes by Determinants
- ### 6. Eigenvalues and Eigenvectors
	- 6.1 Introduction to Eigenvalues: $Ax=Ax$
	- 6.2 Diagonalizing a Matrix
	- 6.3 Symmetric Positive Definite Matrices
	- 6.4 Complex Numbers and Vectorsand Matrices
	- 6.5 Solving Linear Differential Equations
- ### 7. The Singular Value Decomposition (SVD)
	- 7.1 Singular Values and Singular Vectors
	- 7.2 Image Processing by Linear Algebra
	- 7.3 Principal Component Analysis (PCA by the SVD)
- ### 8. Linear Transformations
	- 8.1 The Idea of a Linear Transformation
	- 8.2 The Matrix of a Linear Transformation
	- 8.3 The Search for a Good Basis
- ### 9. Linear Algebra in Optimization
	- 9.1 Minimizing a Multivanable Function
	- 9.2 Backpropagation and Stochastic Gradient Descent
	- 9.3 Constraints, Lagrange Multipliers, Minimum Norms
- ### 10. Learning from Data
	- 10.1 Piecewise Linear Learning Functions
	- 10.2 Creating and Experimenting
	- 10.3 Mean, Variance, and Covariance
# Preface
[MIT 18.06 Linear Algebra Youtube](https://www.youtube.com/playlist?list=PLE7DDD91010BC51F8)
## Introduction
두 column vector $a_1, a_2$ 와 $\text{zero vector}$를 가정하고 시작하자. \
이들은 3차원 공간의 한 점에 대응한다.

이 두 vector를  $\text{Linear Combination}$을 사용하여 나타낼 수 있으며, 이 combination들은 한 평면을 모두 차지한다. \
3차원 공간상의 무한한 평면이다.
$$
	\text{Linear Combination} = ca_1 + da_2 \; \text{ for any numbers c and d}
$$
![[preface_0.png]]

이제 linear algebra의 fundamental idea가 등장한다 : **a matrix** \
Matrix $A$ 는 n개의 column vectors $a_1, a_2, a_3, ..., a_n$ 을 가지며, 위의 경우 3차원 공간상의 두 column vector $a_1, a_2$ 를 의미한다. \
따라서, $A$는 3개의 row($dimension$)와 2개의 column($n$)을 가진다. 
$$
	A = 
	\begin{bmatrix}
	a_1 & a_2
	\end{bmatrix} =
	\begin{bmatrix}
	2 & 1 \\
	3 & 4 \\
	1 & 2
	\end{bmatrix}
$$
3차원 공간상의 2개의 column의 combination들은 plane을 생성하며, 이를 matrix의 **column space**라고 부른다.
모든 $A$에 대해서, $A$의 column space는 column의 모든 combination들을 포함한다.
곧 소개될 Chapter 1에서 볼 4가지 idea들이다.

> [!NOTE] Ideas
> 1. **Column vectors** $a_1$ and $a_2$ in 3 dimensions
> 2. **Linear Combinations** $ca_1+da_2$ of those vectors
> 3. **The matrix $A$** contains the columns $a_1$ and $a_2$
> 4. **Column space of the matrix** = all linear combinations of the columns = plane

이제 $A$가 2개 이상의 column을 가질 때를 생각해보자.
$$
	A = 
	\begin{bmatrix}
	2 & 1 & 3 & 0 \\
	3 & 4 & 7 & 0 \\
	1 & 2 & 3 & -1
	\end{bmatrix}
$$
Linear algebra는 모든 column space를 이해하는 것을 목적으로 하며, \
이를 행렬 $A$에 대해서도 시도해보자.
- Column 1과 2는 이전과 같은 평면을 이룬다.
- Column 3 $a_3 = a_1 + a_2$ 이므로 해당 plane에 있고, 새로운 기여를 하지 않는다.
- Column 4 는 해당 plane에 있지 않다: $c_4a_4$ 는 plane을 높이거나 낮춘다.\
	행렬 $A$의 column space는 전체 3차원 공간이다!

각각의 column은 서로 independent 할 수 있고, combination이 될 수 있다. \
3차원의 모든 점을 생성하기 위해서는, 3개의 independent한 column이 필요하다.

## Matrix Multiplication $A=CR$
"Linear combination"과 "independent columns"는 $3 \times 4$ matrix $A$를 잘 표현한다. \
Column 3은 linear combination이며, Column 1, 2, 4는 independent하다. \
Matrix $C$의 column은 $A$의 independent한 column들을 뽑고, \
Matrix $R$의 column은 $A$의 column을 생성하는 $C$의 combination들을 알려준다.
$$
	A =
	\begin{bmatrix}
	2 & 1 & 3 & 0 \\
	3 & 4 & 7 & 0 \\
	1 & 2 & 3 & -1
	\end{bmatrix} =
	\begin{bmatrix}
	2 & 1 & 0 \\
	3 & 4 & 0 \\
	1 & 2 & -1
	\end{bmatrix}
	\begin{bmatrix}
	1 & 0 & 1 & 0 \\
	0 & 1 & 1 & 0 \\
	0 & 0 & 0 & 1
	\end{bmatrix} =
	CR
$$
## Matrix Multiplication: Each column $j$ of $CR$ is $C$ times column $j$ of $R$
Section 1.3에서는 "$\text{a matrix} \times \text{a vector}$" 에 대해 다루며, \
section 1.4에서는 "$\text{a matrix} \times \text{a matrix}$" 에 대해 다룬다. \
이들은 linear algebra의 중요한 operation들이다. \
이 곱셈을 할 수 있는 좋은 방법이 여러 가지 있다는 것이 중요하다.

Preface의 목적은 큰 그림을 그리는 것이므로, 여기까지만 다룰 것이다. \
다음 페이지에서는 이 주제를 구성하는 두 가지 방법을 알려준다. \
특히 대부분의 linear algebra 과정을 채우는 첫 일곱 개의 장을 소개한다. \
그런 다음 선택적 장이 제공되며, 오늘날 응용 분야에서 가장 활발한 deep learning이라는 주제로 이어진다.

## The Four Fundamental Subspaces
지난 챕터에서, Matrix $A$ 로부터 2가지 step을 시행했다.
1. 첫번째 step은 모든 column의 조합 $ca_1 + da_2 + ea_3 + fa_4$ 을 취하고, 이를 통해 column space를 생성하는 것이다.
2. 두번째 step은 matrix $A$를 $C$ 곱하기 $R$로 인수분해하는 것이고, 이 matrix $C$ 는 전체 independent column 집합을 가지고 있다.

모든 matrix는 4개의 fundamental subspace를 가지고 있다. \
A의 column space 뿐 만 아니라, 모든 row combination의 **row space**도 존재한다. \
n개의 column과 m개의 row의 모든 combination을 취할 때, 이러한 combination들은 vector의 "space"를 채운다.

다른 2개의 subspace가 그림을 완성시킬 것이다. \
Row space가 3 차원 plane이라고 가정해보자. \
그러면 3D 그림에서 한개의 특별한 방향이 있는데, 그 방향은 row space에 perpendicular하다. \
이 perpendicular line은 matrix의 **null space**를 의미한다. \
우리는 모든 row에 perpendicular한 null space의 vector들이 가장 기본적인 linear equation인 $Ax = 0$을 푸는 것을 보게 될 것이다. \
그리고, 모든 row에 perpendicualr인 vector가 중요하다면 모든 column에 perpendicular인 vector도 중요할 것이다. \
다음은 위에서 설명한 4가지의 fundamental subspace의 그림이다.

![[preface_1.png]]

이 그림은 Chapter 3에서 다시 보게 되며, perpendicular space에 대한 idea는 Chapter 4에서 발전시킨다. \
그리고 4개의 subspace에 대한 특별한 "basis vectors"는 Chapter 7에서 다시 다루게 된다. \
그 단계는 fundamental theorem of linear algebra의 마지막 부분이다. \
이 theorem에는 정사각형 또는 직사각형 matrix에 대한 놀라운 사실이 포함되어 있다: independent column의 수는 independent row의 수와 같다.

## Five Factorizations of a Matrix
다음은 linear algebra의 organizing principles이다. \
Matrix가 특별한 속성을 가질 때 이러한 factorization이 이를 보여준다. \
장이 끝나면 핵심 아이디어를 직접적이고 유용한 방식으로 표현한다.

목록을 내려갈수록 유용성이 증가한다. \
Orthogonal matrix는 column이 perpendicualr unit vector이기 때문에 결국 승자가 된다. \
이것이 완벽함이다.
$$
	\text{2 by 2 Orthogonal Matrix} = 
	\begin{bmatrix}
	\cos\theta & -\sin\theta \\
	\sin\theta & \cos\theta
	\end{bmatrix} =
	\text{Rotation by Angle}\;\theta
$$

> [!Tip] Why orthogonal matrix is perfection?
> 저자가 이런 표현을 쓴 이유는 orthogonal vector의 여러 성질 때문일 것이다. \
> Orthogonal matrix에 대해서는 이후에 자세히 설명하겠지만, 항상 linearly independent하다, transpose와 inverse가 같다, projection의 단순해진다 등의 여러가지 성질 덕분에 orthogonal matrix는 수학적으로 아름답고 실용적으로도 매우 유용한 "완벽한" 행렬로 여겨진다.

다음은 Chapter 1, 2, 4, 6, 7의 5가지 factorization이다.
$$
	A = CR = R \text{ combines independent columns in } C \text{ to give all columns of } A
$$
$R$은 $C$의 independent column을 combine하여 $A$의 모든 column을 제공.
$$
	A=LU = \text{Lower triangular } L \text{ times Upper triangular } U
$$
$A$를 lower triangular matrix $L$과 upper triangular matrix $U$으로 factorization.
$$
	A=QR = \text{Orthogonal matrix } Q \text{ times Upper triangular } R
$$
$A$를 orthogonal matrix $Q$와 upper triangular matrix $R$으로 factorization.
$$
	S = Q \Lambda Q^T = \text{(Orthogonal Q) (Eigenvalues in }\Lambda\text{) (Orthogonal }Q^T\text{)}
$$
Symmetric matrix S를 orthogonal matrix Q와 $\Lambda$의 Eigenvalue로 factorization, diagonalization.
$$
	 A = U\Sigma V^T = \text{(Orthogonal U) (Singular values in }\Sigma\text{) (Orthogonal }V^T\text{)}
$$
$A$를 orthogonal matrix $U$, $V$와 $\Sigma$의 Singular value로 factorization, decomposition.

특히 7장의 $A = U\Sigma V^T$ 는 **Singular Value Decomposition(SVD)** 라고 하며, 모든 matrix A에 대해 적용가능하다. \
$U$와 $V$는 모두 길이가 1인 perpendicular column을 가지고 있다. \
Vector에 $U$ 또는 $V$를 곱하면 동일한 길이의 vector가 남기 때문에 계산이 크게 증가하거나 감소하지 않는다. \
그리고 $\Sigma$는 sigular value의 positive diagonal matrix이다. \
Chapter 6에서 eigen values와 eigen vectors에 대해 배운 후, Chapter 7.1에서 singular value에 대해 다룰 것이다.

## Deep Learning
선형 대수학의 진정한 그림을 그리기 위해서는 응용 프로그램이 포함되어야 한다. \
Completenesss은 완전히 불가능할 것이다. \
현재 응용 수학의 지배적인 방향에는 한 가지 특별한 요구 사항이 있다: 완전히 linear일 수는 없다!

그 방향 중 하나가 "deep learning" 이다. \
그것은 fundamental한 과학적 문제에 대한 매우 성공적인 접근 방식이다: **Learning from data**. \
많은 경우에서, data는 matrix 형태로 표현이 가능하다. \
우리의 목표는 variables 간의 연결을 찾기 위해 matrix 내부를 살펴보는 것이다.  
Matrix equation이나 알려진 input-output rule을 표현하는 differential equation을 푸는 대신, 우리는 그 규칙들을 찾아야 한다. \
딥러닝의 성공은 두 가지 종류의 입력 $x$와 $v$를 가진 함수 $F(x, v)$를 구축하는 것이다:
- Vector v는 training data의 feature를 나타낸다.
- Matrix x는 해당 feature들에 weight를 할당한다.
- Function $F(x, v)$는 해당 training data $v$에 대한 올바른 출력에 가깝습니다.
