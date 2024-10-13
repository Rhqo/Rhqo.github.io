![[Two-Stream Convolutional Network_0.png]]

- Spatial stream Convolution
    
    각 비디오 프레임의 정적 정보만으로도 동작을 인식 가능
    
- Temporal stream Convolution
    
    시간에 따른 동작 정보를 인식하기 위해 dense optical flow을 사용하여 프레임 간의 움직임을 분석하고, 이를 ConvNet에 학습