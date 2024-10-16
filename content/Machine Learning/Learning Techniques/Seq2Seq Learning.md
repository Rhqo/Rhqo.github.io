Sequence-to-Sequence Learning은 서로 연관된 두 개의 데이터 시퀀스 간의 관계를 모델링하는 방식

어떤 데이터 시퀀스가 있고, 또 다른 축의 데이터 시퀀스가 있을 때, 그 두 개의 데이터 사이에 어떤 연관 관계나, 어떤 입력이 들어가면 어떤 출력이 나와야 한다는 원인과 결과 관계를, 이 두 데이터 사이에서 모델링을 통해 구현할 수 있는 학습 방식이 있다면, 그것을 sequence-to-sequence learning이라고 부릅니다.

입력 시퀀스를 보통 x라고 하고, 출력을 y라고 한다. 즉, 시퀀스 x가 있고, 다른 데이터에 대응하는 시퀀스 y가 있습니다. 입력 시 시퀀스의 아이템 개수를 N이라 하고, 출력 시 데이터의 개수를 M이라 해보면, N과 M의 숫자에 따라서 여러가지 케이스로 나눌 수 있다.

**Case : N과 M 이 같은 경우**  
시퀀스의 아이템 개수(N)와 출력 데이터의 개수(M)가 완전히 동일한 경우. 
N개의 데이터가 입력되면 N(=M)개의 데이터가 출력되는 형태

**Case : N과 M 이 다른 경우  
- Case N21  
    : 복수의 입력(N)에 대해 딱 하나의 출력(M=1)만 나오는 특별한 경우. 
    M=1라는 것은, 복수의 입력에 대해 단일 출력 결과를 생성한다는 것을 의미한다.
    대표적으로 text classification 같이 복수의 토큰을 입력으로 받아 하나의 class label 이 출력으로 나오는 경우가 이에 해당한다.
- Case N2M  
    : 입력 시퀀스의 아이템 개수(N)와 출력 데이터의 개수(M)가 다른 경우. 
    6개의 데이터가 입력되었을 때 7개 또는 2개가 출력되는 등의 다양한 경우가 있을 수 있다. 
    예를 들어, 번역문제 같은 경우가 입력되는 문장의 토큰 수와 출력의 토큰 수가 다른 경우가 있다.