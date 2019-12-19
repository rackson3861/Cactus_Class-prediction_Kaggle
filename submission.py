import pandas as pd

df1 = pd.read_csv('absent_test.csv')
df2 = pd.read_csv('present_test.csv')
df3 = pd.read_csv('sample_submission.csv')


for i in range(df1.shape[0]):
    df3.loc[df3['id']==df1.iloc[i,0],'has_cactus'] = df1.iloc[i,1]
    
for i in range(df2.shape[0]):
    df3.loc[df3['id']==df2.iloc[i,0],'has_cactus'] = df2.iloc[i,1]
    
df3.to_csv('submission.csv',index = False)