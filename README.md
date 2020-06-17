# 내 손안의 약국
EWHA CSE 졸업프로젝트 



### 프로젝트 소개







### Dataset

: ![img/data.png](https://lh5.googleusercontent.com/6KXleiAT1S5vyYVen3rw__xaQovIlKSczM9QQ1ja3chD_LySSG-exdOYuA54ZyC4znanV5iUHW541z_TdaUwChqlj5mPqAk9WSVtHlrAhzcBB40lFRUvxqaZsEhThhJg)



 : 총 200여종의 알약을 수집

 : 다양한 촬영조건에서 찍은 알약 데이터를 얻기 위해 같은 알약을 크기/밝기/배경/촬영각도를 조절하여 스마트폰으로 촬영

: 현재까지 80종의 알약을 촬영 & 약 2000개의 알약데이터를 수집



### Labeling

: [labelimg tool](https://github.com/tzutalin/labelImg) 사용하여 labeling 진행

: 크게 두가지 방식

1. **로고/분할선/문자 3가지 라벨로 라벨링**

   ![](img/labeling1.png)



2. **전체 식별 문자를 "box" - 1개의 라벨로 라벨링**





### 진행상황

### - text detection model

#### 1. YOLO

![img/yolo_result.png](img/yolo_result.png)



#### 2. EAST

![](img/east_result1.png)

![](img/east_result2.png)



### - text recognition model

#### 1. CRNN

![](img/crnn_result.png)

  : 알약 데이터로 학습하지 않은 CRNN demo 모델로 Text Recognition을 수행

  : 기본적인 text는 잘 인식하지만, 흘림체로 적힌 경우 제대로 인식하지 못하며, 회사로고와 같은 기호를 text로 인식한다는 문제점이 있음