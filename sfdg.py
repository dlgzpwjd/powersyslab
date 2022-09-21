import pandas as pd
raw_data = {'col0' : [1, 2, 3, 4],
             'col1' : [10, 20, 30, 40],
              'col2' : [100, 200, 300, 400]} #리스트 자료형으로 생성 
raw_data1 = pd.DataFrame(raw_data) #데이터 프레임으로 전환 및 생성
raw_data2 = pd.DataFrame(raw_data) #데이터 프레임으로 전환 및 생성
xlxs_dir='sample.xlsx' #경로 및 파일명 설정
with pd.ExcelWriter(xlxs_dir) as writer:
    raw_data1.to_excel(writer, sheet_name = 'raw_data1') #raw_data1 시트에 저장
    raw_data2.to_excel(writer, sheet_name = 'raw_data2')