# Kline prediction and trade in Okex 

## Summary 
- **Data**  
x_feature(multi period tech factors):  
ma1,ma2…,rsi1,rsi2……  
y_label(classify by abs(pct))  
[-2, -1, 0 , 1, 2]  

- **Model**  
RandomForestClassifier

- **Trade**  
```
if predict_prob > threshold:  
   if pred_y_label > 0:  
      open_long  
   elif pred_y_label < 0:  
      open_short  
   else:  
      close_all 
else:
   close_all
```
## Setup
- **Linux**:    
bash Strategy/app/run_okex_strategy.sh &

- **Windows**:    
python Strategy/app/okex_strategy.py
