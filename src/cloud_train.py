#!/usr/bin/env python
'''
Recommendations via Spark ALS Matrix Decomposition. Deploy on Google Cloud as:

spark-submit cloud_train.py [CLOUDSQL_INSTANCE_IP] [CLOUDSQL_DB_NAME] [CLOUDSQL_USER] [CLOUDSQL_PWD]

Summary of Everest Law's edits to the original file
    - Added vital comments and docstrings
    - Persist repeatedly-used large RDDs
    - ALS.train() -> ALS.trainImplicit(), since this project works only with implicit ratings

Original file from https://github.com/GoogleCloudPlatform/spark-recommendation-engine
Original licence declaration below.

    ===
    Copyright Google Inc. 2016
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    ===
'''

import sys
import itertools
from operator import add
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

from pyspark import StorageLevel

#[START read_data]
'''
    Rating data is stored on SQL server; supply details through command-line
'''
CLOUDSQL_INSTANCE_IP = sys.argv[1]
CLOUDSQL_DB_NAME = sys.argv[2]
CLOUDSQL_USER = sys.argv[3]
CLOUDSQL_PWD  = sys.argv[4]

TABLE_RATINGS = "Rating"
TABLE_RECOMMENDATIONS = "Recommendation"

# JDBC URL that Spark will use to read from SQL server
jdbcUrl = 'jdbc:mysql://%s:3306/%s?user=%s&password=%s' % (CLOUDSQL_INSTANCE_IP, CLOUDSQL_DB_NAME, CLOUDSQL_USER, CLOUDSQL_PWD)

# Read the data from the Cloud SQL
# Create dataframes
dfRates = sqlContext.read.jdbc(url=jdbcUrl, table=TABLE_RATINGS)

# Cache to avoid excessive rereads
rddRates = dfRates.rdd
rddRates.persist(StorageLevel.MEMORY_AND_DISK)
print("Training on {} user-song pairs".format(rddRates.count()))
#[END read_data]


#[START train_model]
'''
    Hyperparameters for matrix decomposition
    Taking defaults from the classic paper
        "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren & Volinsky, 2008)

    Ideally tuned by cross-validation, offline, before deploying system (PENDING)
'''
finalRank  = 40     # Paper tried ranks of 10^1 -10^2; 40+ -> diminishing returns
finalRegul = 0.1
finalIter  = 10     # Spark docs: ALS 'typically' converges within 20 iterations
finalAlpha = 40     # Paper: acceptable value from experiment on movie data

model = ALS.trainImplicit(rddTraining, finalRank, finalIter, float(finalRegul), alpha=finalAlpha)
#[END train_model]


#[START save_to_DB]
'''
    Generate predictions for all unrated items, then save to DB
'''
potentialPairs = dfRates.select('uid').cartesian(dfRates.select('song_id'))
unratedPairs = dfRates.join(potentialPairs, on=['uid', 'song_id'], how='left').select('uid', 'song_id')
predictions = model.predictAll(unratedPairs.rdd).lambda(p: Row(uid=p[0], song_id=p[1], rating=p[2]))

dfToSave = sqlContext.createDataFrame(topPredictions)
dfToSave.write.jdbc(url=jdbcUrl, table=TABLE_RECOMMENDATIONS, mode='overwrite')
#[END save_to_DB]
