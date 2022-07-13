## Nueral Network
위성 자료를 이용하여 지면 초미세먼지(PM2.5) 농도를 산출하는 예제. ANN과 CNN의 기본 모델을 사용하여 농도를 산출하는 코드를 작성할 수 있음.  
이를 위해 본 파일에서는 데이터셋을 이용하여 ANN과 CNN 모델을 구축하는 과정을 서술. 

###	1) nc_ref.py
: 데이터를 전처리하고 ANN 혹은 CNN 신경망을 적용하는 코드
#### 코드 설명: 아래 제목은 #%%로 분리되어 있음. 
0. define function    
-open_netcdf(): netcdf4 파일을 읽는 함수  
-ZNormScaler(): {(값-평균)/표준편차}를 계산하여 Z 점수를 계산하는 함수  
-choice_train_test_split(): 지정된 비율에 맞추어 데이터셋을 분리하는 함수. shuffle 인자를 true로 두면 무작위로 인덱싱하여 분리함.   
1. NN SETTING  
-신경망 구축을 위한 변수를 정의.  
2. DATA PREPARATION  
-일별 데이터로부터 gems_rad, gems_geo, 그리고 8개의 기상변수를 X_raw, pm25를 Y_raw로 저장. 사이즈를 일치시키기 위해 transpose를 하고, vstack()으로 연결함.   
3. PRE-PROCESSING DATA   
-이상치 제거를 위해 Q1값과 Q3 값을 percentile()로 계산하여 범위 밖의 값을 제거.   
-사전에 정의한 ZNormScaler()로 Z 점수를 계산하여 X_tr로 저장.   
4. PCA  
-주성분분석 PCA (Principal Component Analysis)를 적용하여 의존성 높은 변수를 결합하고 데이터 크기를 감소시켜 학습 속도를 높이고 과적합을 방지.   
-기존의 365개 변수를 90개 변수로 감소시킴.   
5. DATA SPLIT  
-앞서 정의한 choice_train_test_split()을 이용해 전체 데이터셋을 train:val:test = 8:1:1로 분리함.   
6. SAVING DATA  
-dump()로 pickle 데이터를 저장  
Option. Oversampling  
-오버샘플링을 적용하는 smogn.smoter()로 극단 값의 양을 증가시킴. 데이터 크기가 큰 만큼 속도가 느리고, 성능을 크게 향상시키지도 않아 적용 여부는 선택.   
7. DATA LOADING  
-load()로 pickle 데이터 로드  
8. MODEL BUILDING AND RUNNING  
-ANN과 CNN의 하이퍼파라미터를 최적화하고 모델을 구축.  
-각 신경망의 레이어를 add()로 연결하고, 가중치 학습 내용을 history에 저장.   
-하이퍼파라미터의 범위를 지정하여 RandomizedSearchCV()를 통해 하이퍼파라미터를 랜덤하게 적용하여 모델 성능을 비교함.  
가장 높은 성능을 가진 파라미터는 Random.best_params_로 접근 가능.   
-최적 파라미터 탐색 후 모델을 complie하고, 훈련 데이터로 fit 진행. 모델은 pickle 데이터로 저장됨.   
9. ANALYSIS  
-plot_history() 함수를 정의하여 history 변수의 손실함수를 시각화함.   
-savefig()로 이미지 저장 가능.  
![image](https://user-images.githubusercontent.com/58411517/178696331-9c54aaf2-a1c0-48f7-b2ae-a00cbd9b38ff.png)
[ANN 손실함수]  
![image](https://user-images.githubusercontent.com/58411517/178696372-2955c846-04d6-4b74-a095-467b05010ccc.png)
[CNN 손실함수]

10. evaluation   
-결과치의 R2를 계산하고, density plot을 그려 참값과 예측값의 일치 정도를 확인. 
![image](https://user-images.githubusercontent.com/58411517/178696449-bd22d05a-41f9-486d-afb7-bf9b49735dbb.png)
[ANN density plot]  
-R2: 0.674, RMSE: 10.4
![image](https://user-images.githubusercontent.com/58411517/178696515-d913b6a3-ce61-4127-9773-e1f782156e0f.png)
[CNN density plot]  
-R2: 0.691, RMSE: 11.3  
=> R2 측면에서는 CNN이, RMSE 측면에서는 ANN이 더 좋은 성능을 보임. 

11. pair-wise analysis    
-scatter()로 참값과 예측값의 차이, 참값, 예측값에 대한 scatter plot을 그림.   
-density plot을 그릴 때는 gaussian_kde로 데이터의 밀도를 구한 후 이를 scatter()로 그림.  


12. error analysis    
-양의 오차, 음의 오차, 오차 범위가 크지 않은 경우로 나누어 변수와의 관계를 파악하기 위한 히스토그램을 그림.   
![image](https://user-images.githubusercontent.com/58411517/178696938-b53b60f1-6cf9-49ae-8ca0-a1909a31a2e6.png)
[ANN 음의 오차 비교]



13. explicit comparison    
-한반도 지도 상에 데이터 비교  
14. estimates line plot    
-일별, 시간별, 일-시간별 PM2.5 농도 참값과 예측값을 비교하는 선형그래프를 그림.   

