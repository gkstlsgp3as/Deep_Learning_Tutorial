## Preparing Datasets
위성 자료를 이용하여 지면 초미세먼지(PM2.5) 농도를 산출하는 예제. ANN과 CNN의 기본 모델을 사용하여 농도를 산출하는 코드를 작성할 수 있음.  
이를 위해 본 파일에서는 필요한 데이터셋을 마련하는 과정을 서술함.

###	1) conv_airkorea.py
: 에어코리아(airkorea.or.kr)에서 다운로드한 측정자료(.xlsx)를 CSV 파일로 변환하는 코드  
(엑셀 파일은 용량이 크고 파일 로드 속도가 느려 CSV 파일로 변환하는 과정이 필요함.)   
에어코리아 사이트에서 [통계정보 > 최종확정 측정자료 조회 > 확정자료 다운로드]로 진행하면 최종확정자료를 연도별로 다운로드할 수 있음.

![image](https://user-images.githubusercontent.com/58411517/178409258-d4bc624a-cdd3-4baa-9e1e-225e38a31b4c.png)  
[참고: 월별 엑셀 파일 예시]

### 2) read_pm25.py
: 1)에서 변환한 파일을 통합하는 코드  
기본적으로 파일은 연도-월별로 구분되어 있으며,  
지역, 망, 측정소코드, 측정소명, 측정일시, SO2, CO, O3, NO2, PM10, PM25, 주소 데이터를 포함하고 있음.   
측정소의 위경도 자료가 없으므로 이를 통합해야 함.   
에어코리아 사이트에서 [통계정보 > 대기환경 월간/연간 보고서 > 대기환경 연보/월보 다운로드]로 진행하여 파일 버튼을 클릭하면  
아래와 같이 파일이 다운로드되고, ‘4. 대기오염측정망 제원’ 파일에서 측정소 별 위경도 좌표를 확인할 수 있음. 

![image](https://user-images.githubusercontent.com/58411517/178409567-8a55cd99-f03d-44eb-b54e-ae50499f8b36.png)  
[참고: 21년 3월 대기환경 월보 압축파일 목록]  
이로부터 측정소 코드와 위경도 좌표 만을 포함한 st_loc.csv 파일을 만듦.  

#### 코드 설명: 아래 제목은 #%%로 분리되어 있음.   
+	read csv files  
-read_csv()로 파일을 읽음. 한글이 포함되어 있으므로 ‘unicode’로 읽어야 함.   
-for문 내에서 파일을 읽으며, 각 측정소 종류 별 개수를 출력함.   
+	read csv files in March  
-21년 3월에 해당하는 csv파일을 읽은 후, 1일부터 31일까지 for문 내에서 측정소 종류 별 개수를 출력함.   
+	preprocessing data in 2021  
-2021년의 월별 파일을 모두 읽으며, 각 파일을 열 때 nan 값이 포함된 열을 dropna()로 삭제함.  
-‘측정일시’ 열이 datetime 타입과 같이 00시에서 23시가 아닌, 01부터 24시까지로 입력되어 있기 때문에 24시의 경우 datetime 타입으로     형변환이 되지 못함. 따라서 해당 경우를 er_ind로 인덱싱하고, 다음 월 00시로 변경하는 과정을 거침. 이후 datetime 타입으로 변환함.   
-2021년 월별 데이터를 concat()으로 합쳐 ‘2021_PM25.csv’ 파일로 통합함.    
+	load data   
-‘2021_PM25.csv’ 파일을 읽고, 파일을 읽을 때 변수가 object 타입으로 읽히기 때문에   
to_numeric()이나 to_datetime()으로 형변환하여 읽음.   
-측정소 위경도 정보가 포함된 ‘st_loc.csv’ 파일을 읽음.   
+	pre-process network  
-측정소 종류가 ‘도시대기’, ‘교외대기’와 같이 한글로 저장되어 있어 처리가 곤란하므로 0, 1, 2와 같이 숫자로 변환.  
+	Insert coordinates of stations  
-앞의 st_loc.csv를 읽은 st_df 데이터프레임을 2021_PM25.csv로 얻은 df 데이터프레임과 merge()로 합침.  
측정소코드를 기준으로 합칠 수 있음.   
+	split into monthly data  
-2021년 전체 데이터가 크므로 월별로 분리   

### 3) nc_avg.py
: Z 점수 정규화를 위한 데이터 평균 및 표준편차를 계산하는 코드 
#### 코드 설명: 아래 제목은 #%%로 분리되어 있음. 
+	define function  
-open_netcdf() 함수를 정의   
-GEMS 데이터가 netcdf4 형태로 저장되어 있으므로 Dataset()로 읽음.   
+	calculate mean and std  
-2021년 3월 파일에서 일별로 gems_rad, gems_geo, pm25 변수, ugrd, vgrd와 같은 기상 변수를 읽은 후   
mean()과 std()를 이용해 axis=1, 즉, 열 단위로 평균 및 표준편차를 계산하여 pickle 파일로 저장.   


