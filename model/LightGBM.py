import lightgbm as lgb
import pandas as pd

def lgbm_train(df, cols=["hr_average","rmssd", "temperature_delta","pss4_1", "breath_average"],trg='pss4_1',train_ratio=0.8,valid_ratio=0.1,test_ratio=0.1):
    #make dataframe for training
    lgbm_df = df[cols]
    tr,vd,te = [int(len(lgbm_df) * i) for i in [train_ratio, valid_ratio, test_ratio]]
    #import pdb; pdb.set_trace()
    X_train, Y_train = lgbm_df[0:tr].drop([trg], axis=1), lgbm_df[0:tr][trg]
    X_valid, Y_valid = lgbm_df[tr:tr+vd].drop([trg], axis=1), lgbm_df[tr:tr+vd][trg]
    X_test = lgbm_df[tr+vd:tr+vd+te+2].drop([trg], axis=1)
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_valid = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)
    #model training
    params = {
        'task' : 'train',
        'boosting':'gbdt',
        'objective' : 'regression',
        'metric' : {'rmse'},
        'num_leaves':200,
        'drop_rate':0.03,
        'learning_rate':0.1,
        'seed':0,
        'feature_fraction':1.0,
        'bagging_fraction':1.0,
        'bagging_freq':0,
        'min_child_samples':5
    }
    gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_train, lgb_valid], early_stopping_rounds=100)
    #make predict dataframe
    pre_df = pd.DataFrame()
    pre_df[trg] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    pre_df.index = lgbm_df.index[tr+vd:tr+vd+te+2]
    return pre_df, gbm, X_train