import pandas as pd
import numpy as np
from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise.reader import Reader
from surprise.prediction_algorithms import knns, BaselineOnly


# Import the csv
user_artist_df = pd.read_csv('user_artist.csv')
test_df = pd.read_csv('test.csv')

test_df['weight'] = 0

minP = user_artist_df.weight.min()
maxP = user_artist_df.weight.max()

reader = Reader(rating_scale=(minP,maxP))
user_artist_data=Dataset.load_from_df(user_artist_df[['userID', 'artistID', 'weight']], reader=reader)

# Split to train and test to train the model
train, test = train_test_split(user_artist_data, test_size=0.2)

# KNN model
sim_cos = {'name':'cosine', 'user_based':False}
knnbasic = knns.KNNBasic(k=40, min_k=1, sim_options=sim_cos)
knnbasic.fit(train)
predictions_sim = knnbasic.test(test)

# Base Line model
bsl_options = {'method': 'als',
               'n_epochs': 50
               }
baseline = BaselineOnly(bsl_options=bsl_options)
baseline.fit(train)
predictions_baseline = baseline.test(test)

# Sum the predictions according to the relevant formula
predictions = [0] * len(predictions_sim)
for i in range(len(predictions_sim)):
    predictions[i] = (predictions_baseline[i].est + predictions_sim[i].est)

# Calculate the Loss
loss = 0
for i in range(len(test)):
     loss += ((np.log10(predictions[i]))-(np.log10(test[i][2])))**2

# Predict on the real test data
rui_hat = []
for i in range(test_df.shape[0]):
    knnPrediction = knnbasic.predict(uid=test_df.iloc[i].userID, iid=test_df.iloc[i].artistID, verbose=True)
    baselinePrediction = baseline.predict(uid=test_df.iloc[i].userID, iid=test_df.iloc[i].artistID, verbose=True)
    rui_hat.append(knnPrediction.est + baselinePrediction.est)

test_df['weight'] = rui_hat
