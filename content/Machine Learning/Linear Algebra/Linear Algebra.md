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
두 column 벡터 $a_1, a_2$ 와 $\text{zero vector}$를 가정하고 시작하자. \
이들은 3차원 공간의 한 점에 대응한다.

이 두 벡터를  $\text{Linear Combination}$을 사용하여 나타낼 수 있으며, 이 combination들은 한 평면을 모두 차지한다. \
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
Section 1.3에서는 $\text{a matrix} \times \text{a vector}$ 에 대해 다루며, \
section 1.4에서는 $\text{a matrix} \times \text{a matrix}$에 대해 다룬다. \
이들은 linear algebra의 중요한 operation들이다. \
이 곱셈을 할 수 있는 좋은 방법이 여러 가지 있다는 것이 중요하다.

Preface의 목적은 큰 그림을 그리는 것이므로, 여기까지만 다룰 것이다. \
다음 페이지에서는 이 주제를 구성하는 두 가지 방법을 알려준다. \
특히 대부분의 linear algebra 과정을 채우는 첫 일곱 개의 장을 소개한다. \
그런 다음 선택적 장이 제공되며, 오늘날 응용 분야에서 가장 활발한 deep learning이라는 주제로 이어진다.

## The Four Fundamental Subspaces
