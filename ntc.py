import pandas as pd 
import numpy as np 
  
# create a dummy array 
arr = np.arange(1,11).reshape(2,5) 
print(arr) 
  
# convert array into dataframe 
DF = pd.DataFrame(arr) 
  
# save the dataframe as a csv file 
DF.to_csv("data1.csv")

