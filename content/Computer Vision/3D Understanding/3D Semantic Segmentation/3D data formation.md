---
title: 3D data formation
---
2D Semantic Segmentation은 2D image에 대해서 수행했기 때문에, "Image"라는 형식에 대한 이해가 필요했다. \
"Image" 형식은 pixel로 구성된 grid형식의 데이터라는 특징이 있다.

하지만 3D Semantic Segmentation은 3D 데이터에 대해서 수행하기 때문에, 새로운 형식에 대한 이해가 필요하다. \
3D를 표현하는 방법은 point cloud, voxel, mesh, implicit, occupancy net 등 여러가지 있다. \
주로 사용되는 표현 방법은 point cloud 형식인데, 대부분의 3D 데이터는 ground truth를 LiDAR로 취득한 point cloud dataset을 사용했기 때문일 것이다.

Point cloud data는 크게 3가지의 특징을 가지고 있다.
- **Unordered** : Points are **not** on a **regular grid**
- **Unstructured** : Points do **not** carry the **neighboring points' information**
- **Irregular** : Points contain regions with **different densities**

거의 모든 3d 데이터를 처리하기 위한 모델은 이러한 3d point cloud의 특징을 기반으로 하여 설계된다. \
특히 point cloud의 위와 같은 특징 때문에, 이미지에서 사용되던 convolution 기반 모델들을 사용하지 못한다. \
예를 들어, convolution을 사용하려면, point cloud를 voxel기반 표현으로 바꾸어 voxel을 grid처럼 사용하여 3d convolution을 수행하여야 한다.

Point cloud 데이터의 특징 때문에 3D 모델들이 필수적으로 가져야 하는 특징은 permutation invariance 혹은 permutation equivariance한 특징이다.

> [!Tips] Invariance vs Equivariance
> ![[INvsEQ_0.png | center]]
> $$
> 	\text{Definition 1.1}\; (g\text{-invariant}).\: \text{For given } f \text{ and } g, f \text{ is } g\text{-invariant if for all } x, \\  
> 	f(g(x)) = f(x)  
> $$
> $$
>	\text{Definition 1.2}\; (g\text{-equivariant}).\: \text{For given } f \text{ and } g, f \text{ is } g\text{-equivariant if for all } x, \\  
>	f(g(x)) = g(f(x))  
> $$
> - **Invariance**: 입력이 어떤 변환을 받아도, 출력은 변하지 않는 성질
> - **Equivariance**: 입력이 변환되면, 출력도 동일한 방식으로 변환되는 성질

최초로 raw한 point cloud를 입력으로 사용한 PointNet은 입력의 순서에 invariant한, symmetric function을 사용하여 permutation invariance한 특징을 가지도록 설계했다. \
Point Transformer v3는 point cloud를 serialize하여 모델이 permutation equivariance한 특징을 가지도록 설계했다.
