---
title: What is Kernel Regression?
draft: true
comments: true
---

# Linear Regression

Linear regression은 데이터에 잘 approxtimate할 수 있는 함수 $f$를 "linear function으로 가정"하고, \
그 함수의 weight을 찾는 과정이다.

$$
f : W\mathbf x+\mathbf b
$$
$$
\begin{split}
Loss(W, b) &= \frac{1}{n}\sum_i^n(f(\mathbf x_i)-\mathbf y_i)^2 \\
&= \frac{1}{n}\sum_i^n(W\mathbf x_i + \mathbf b-\mathbf y_i)^2
\end{split}
$$

