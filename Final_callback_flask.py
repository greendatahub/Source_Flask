import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import warnings
import joblib
#from sklearn.externals import joblib
warnings.filterwarnings(action='ignore')
from flask import Flask,request

def preprocessing_ML(path,return_date): # return_date 형태는 '2021-01-05', ''포함해 앞과 같은 형태 #매개변수가 모델 경로 지정
    # 데이터 로드
    DF_env = pd.read_excel(path,sheet_name = '환경정보_일별(딸기)')
    DF_growth = pd.read_excel(path,sheet_name = '생육정보_일별(딸기)')
    DF_size = pd.read_excel(path,sheet_name = '재배면적').iloc[0,0]

  # returndate로 첫 수확 날짜=생육측정 날짜를 받으면, 그시기의 2주전 까지의 데이터를 훈련데이터로 사용 
    DF_env['수집일'] = pd.to_datetime(DF_env['수집일'])
    DF_growth['조사일'] = pd.to_datetime(DF_growth['조사일'])
    DF_env=DF_env.set_index('수집일')
    DF_growth=DF_growth.set_index('조사일')
    cut_date = DF_growth[ :return_date].index[-3]
    DF_env = DF_env[cut_date:]
    DF_growth = DF_growth[cut_date:]
    DF_env=DF_env.resample(rule='d').mean()
    DF_env=DF_env.resample(rule='7d',label='left').mean()
    final_DF = pd.concat([DF_growth,DF_env],axis=1)
    final_DF=final_DF.dropna(axis=0)
    final_DF.columns=['초장(cm)','엽장(cm)','엽폭(cm)','엽수(개)','착과수(개)','내부CO2(ppm)','내부습도(%)','내부온도(도)']
    start_date=final_DF.index[-1]

    return final_DF, start_date,DF_size

def preprocessing_ML2(path,start_date): # 이전 작기
    # 데이터 로드
    DF_env = pd.read_excel(path,sheet_name = '환경정보_일별(딸기)') 
    DF_growth = pd.read_excel(path,sheet_name = '생육정보_일별(딸기)')

  # returndate로 첫 수확 날짜=생육측정 날짜를 받으면, 그시기의 2주전 까지의 데이터를 훈련데이터로 사용 
    DF_env['수집일'] = pd.to_datetime(DF_env['수집일'])   + datetime.timedelta(days=365)
    DF_growth['조사일'] = pd.to_datetime(DF_growth['조사일'])  + datetime.timedelta(days=365)

    DF_env=DF_env.set_index('수집일')
    DF_growth=DF_growth.set_index('조사일')
    cut_date = DF_growth[start_date:].index[0]

    DF_env = DF_env[cut_date:]
    DF_growth = DF_growth[cut_date:]
    DF_env=DF_env.resample(rule='d').mean()
    DF_env=DF_env.resample(rule='7d',label='left').mean()
    final_DF = pd.concat([DF_growth,DF_env],axis=1,ignore_index=True)
    final_DF=final_DF.dropna(axis=0)
    final_DF.columns=['초장(cm)','엽장(cm)','엽폭(cm)','엽수(개)','착과수(개)','내부CO2(ppm)','내부습도(%)','내부온도(도)']


    return final_DF

def preprocessing_LSTM(Data,past=3):
    생육데이터=Data[['초장(cm)','엽장(cm)','엽폭(cm)','엽수(개)','착과수(개)']]
    환경데이터=Data[['내부CO2(ppm)','내부습도(%)','내부온도(도)']]  
    train_X1=[]
    train_X2=[]
    n_future = 1 # 에측하고자하는 미래의 날짜 거리
    n_past = past # 에측에 사용하고자하는 환경데이터의 시퀀스의 포함 주차 
    for i in range(n_past, len(환경데이터) - n_future +2):
      train_X1.append(환경데이터.iloc[ i - n_past: i , : ])
      train_X2.append(생육데이터.iloc[ i - n_past: i , : ])
    train_X1,train_X2 = np.array(train_X1),np.array(train_X2)
    return train_X1.astype('float32'), train_X2.astype('float32')

class prediction(object):

  def __init__(self,model_path,scaler_path):

    self.scaler = joblib.load(scaler_path)  
    #self.scaler = joblib.load('/content/drive/MyDrive/2022_그린데이터랩_프로젝트/딸기 데이터/모델저장/all_farm_scaler.pkl')   
    self.model_path = model_path

  def preparing_data(self,data):
    #data.iloc[:,2:-2] = self.scaler.transform(data.iloc[:,2:-2])
    data = pd.DataFrame(data = self.scaler.transform(data), columns = data.columns)
    train_X1, train_X2 = preprocessing_LSTM(data)
    return train_X1, train_X2

  def prediction_output(self,data,length,size,return_date):
    train_X1, train_X2= self.preparing_data(data)
    #loaded = keras.models.load_model("/content/drive/MyDrive/2022_그린데이터랩_프로젝트/딸기 데이터/모델저장/all_Farm_LSTM5.hdf5")
    loaded = keras.models.load_model(self.model_path)
    y_pred=loaded.predict([train_X1,train_X2]).astype('float32')
    y_pred = y_pred*size
    y_pred = y_pred.tolist()
    value = []
    for i in range(len(y_pred)):
        value.append(y_pred[i][0])
    base = datetime.datetime.strptime(return_date,'%Y-%m-%d') + datetime.timedelta(days=7)
    date_list = [base + datetime.timedelta(weeks=x) for x in range(length) if x <= 17]
    date=[]
    for i in date_list:
        date.append(i.strftime("%Y-%m-%d"))
    total_output = {'date': date , 'pred' : value}
    print(total_output)
    return total_output

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def predict(num=None):
    path1 = sys.argv[1]
    path2 = sys.argv[2]
    return_date = sys.argv[3]
    model_path = sys.argv[4]
    scaler_path = sys.argv[5]
    #params = request.get_json()
    #path1 = params['path1']
    #path2 = params['path2']
    #model_path = params['model_path']
    #scaler_path =  params['scaler_path']
    #return_date = params['return_date']
    previous_data, start_date,size = preprocessing_ML(path1,return_date)
    now_data = preprocessing_ML2(path2,start_date)
    final_DF = pd.concat([previous_data,now_data])
    length = len(final_DF)-2
    model = prediction(model_path,scaler_path)
    response = model.prediction_output(final_DF,length,size,return_date)
    return jsonify(response)
# 표준화 전처리 후 preprocessing_LSTM 필요


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)
    #app.run(debug = True)
