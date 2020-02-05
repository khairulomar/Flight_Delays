import pandas as pd

def clean_up(df):
    
    # Clean up null
    df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(df[(df['ORIGIN_AIRPORT']=='EWR') & (df['DEPARTURE_TIME'].isna())].index, inplace=True)
    df.drop(df[(df['DESTINATION_AIRPORT']=='EWR') & (df['ARRIVAL_TIME'].isna())].index, inplace=True)
    df.drop(df[(df['ARRIVAL_DELAY'].isna())].index, inplace=True)
    delay = list(df[df.columns[-6:]].columns)
    for cols in delay:
        if df[cols].isna().any():
            df[cols].fillna(value=0, inplace=True)
    
    # Clean up date format
    df['DATE'] = pd.to_datetime(df[['YEAR','MONTH', 'DAY']])

    # Clean up Arrival time format
    df["HOUR"] = df["SCHEDULED_ARRIVAL"].apply(lambda x: int(str(int(x))[0:2]) if len(str(int(x)))==4 else int(str(int(x))[0:1]))
    df["MINUTE"]  = df["SCHEDULED_ARRIVAL"].apply(lambda x: int(str(int(x))[-2:]))
    df["SCHEDULED_ARRIVAL"] = pd.to_datetime(df[['YEAR','MONTH', 'DAY', 'HOUR', 'MINUTE']])
    df["SCH_ARR_TEMP"] = pd.to_datetime(df[['YEAR','MONTH', 'DAY', 'HOUR']])

    # Clean up Departure time format
    df["HOUR"] = df["SCHEDULED_DEPARTURE"].apply(lambda x: int(str(int(x))[0:2]) if len(str(int(x)))==4 else int(str(int(x))[0:1]))
    df["MINUTE"]  = df["SCHEDULED_DEPARTURE"].apply(lambda x: int(str(int(x))[-2:]))
    df["SCHEDULED_DEPARTURE"] = pd.to_datetime(df[['YEAR','MONTH', 'DAY', 'HOUR', 'MINUTE']])
    df["SCH_DEP_TEMP"] = pd.to_datetime(df[['YEAR','MONTH', 'DAY', 'HOUR']])

    # Temporary column for Darksky data mapping
    df["NYC_TIME_TEMP"] = df.apply(lambda row: row['SCH_ARR_TEMP'] if row['DESTINATION_AIRPORT'] == 'EWR' else row['SCH_DEP_TEMP'], axis=1)

    # Combine United Express (EV) as mainline United (UA)
    df['AIRLINE'] = df.apply(lambda row: 'UA' if row['AIRLINE'] == 'EV' else row['AIRLINE'], axis=1)

    # Simplify delay reason columns - for EDA only, not as predictors
    df['DELAY_REASON'] = df.apply(lambda row: 'Air system' if row['AIR_SYSTEM_DELAY'] == 1 else 
                                  ('Security' if row['SECURITY_DELAY'] == 1 else 
                                   ('Airline' if row['AIRLINE_DELAY'] == 1 else 
                                    ('Late aircraft' if row['LATE_AIRCRAFT_DELAY'] == 1 else 
                                     ('Weather' if row['WEATHER_DELAY'] == 1 else 0)))), axis=1)

    # Dummy variables for Arrival or Departure
    df['DEPARTURE'] = df.apply(lambda row: 0 if row['DESTINATION_AIRPORT'] == 'EWR' else 1, axis=1)

    # Time in reference to Newark only
    df['SCHEDULED_TIME'] = df.apply(lambda row: row['SCHEDULED_DEPARTURE'] if row['DEPARTURE'] == 1 else row['SCHEDULED_ARRIVAL'], axis=1)

    # Time in reference to Newark only - Hour
    df['SCHEDULED_HOUR'] = df['SCHEDULED_TIME'].apply(lambda row: row.hour)

    # Dummy variables for target (Delay)
    df['DELAY'] = df.apply(lambda row: 1 if (row['DEPARTURE']==1)&(row['DEPARTURE_DELAY']>15) else (1 if (row['DEPARTURE']==0)&(row['ARRIVAL_DELAY']>15) else 0), axis=1)
    return df


# Automate Confusion Matrix
import numpy as np
import itertools
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, auc

# Save result of each confusion matrix to be compared between different models later:

def cf_measures(y, y_pred):
    cf_matrix = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = cf_matrix[0][0], cf_matrix[0][1], cf_matrix[1][0], cf_matrix[1][1]
    TPR = round(TP/(TP+FN),4) # Power
    FPR = round(FP/(FP+TN),4) # Alpha
    FNR = round(FN/(FN+TP),4) # Beta
    TNR = round(TN/(TN+FN),4)
    Acc = round((TP+TN)/np.sum(cf_matrix),4)
    Pre = round(TP/(TP+FP),4)
    F1S = round(f1_score(y, y_pred),4)
    return TPR, FPR, FNR, TNR, Acc, Pre, F1S

# Plot confusion matrix and display summary
def cf_matrix(y, y_pred):
    cf_matrix = confusion_matrix(y, y_pred)
    plt.imshow(cf_matrix,  cmap=plt.cm.Blues)
    thresh = cf_matrix.max() / 2.  
    for i, j in itertools.product(range(cf_matrix.shape[0]), range(cf_matrix.shape[1])):
            plt.text(j, i, cf_matrix[i, j],
                     horizontalalignment='center', fontsize=12,
                     color='white' if cf_matrix[i, j] > thresh else 'black')
    plt.ylabel('TRUE Delay')
    plt.xlabel('PREDICTED Delay')
    plt.xticks([0,1])
    
    TPR, FPR, FNR, TNR, Acc, Pre, F1S = cf_measures(y, y_pred)    
    print (f'FalsePos={round(FPR*100,2)}%, FalseNeg={round(FNR*100,2)}%')
    print(f'TruePos={round(TPR*100,2)}%, TrueNeg={round(TNR*100,2)}%')
    print(f'Accuracy={round(Acc*100,2)}%, Precision={round(Pre*100,2)}%')
    print(f'F1score={round(F1S*100,2)}%')
    return ('Confusion Matrix:')

from sklearn.metrics import roc_auc_score
def scores(model,X_train,X_val,y_train,y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    
def annot(fpr,tpr,thr):
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1
        
from sklearn.metrics import roc_curve, auc
def roc_plot(model,X_train,y_train,X_val,y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    plt.figure(figsize=(5,5))
    fpr1, tpr1, threshold1 = roc_curve(y_train, train_prob)
    plt.plot(fpr1, tpr1)    
    fpr2, tpr2, threshold2 = roc_curve(y_val, val_prob)
    plt.plot(fpr2, tpr2)            
    auc_train = round(auc(fpr1, tpr1),4)
    auc_val   = round(auc(fpr2, tpr2),4)    
   #annot(fpr, tpr, threshold)    
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('True Positive (power)')
    plt.xlabel('False Positive (alpha)')
    plt.legend(['Train','Validation'])
    plt.show()
    print(f'AUC train: {round(auc_train*100,2)}%, AUC validation: {round(auc_val*100,2)}%')
    return auc_train, auc_val  
    
def opt_plots(opt_model, index, column):
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index=index,columns=column,values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index=index,columns=column,values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')
#   return opt