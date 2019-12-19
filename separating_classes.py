import cv2
import pandas as pd

df = pd.read_csv('train.csv')

for i in range(df.shape[0]):
    if df.iloc[i,1]==0:
        img = cv2.imread('train/'+df.iloc[i,0])
        cv2.imwrite('class0/'+df.iloc[i,0],img)
        print('a '+ str(i))
    else:
        img = cv2.imread('train/'+df.iloc[i,0])
        cv2.imwrite('class1/'+df.iloc[i,0],img)
        print('b '+ str(i))