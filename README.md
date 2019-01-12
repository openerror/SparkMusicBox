# Music Box Churn Prediction and Recommendation
## Introduction

A study on 1) how to prevent user loss ('churning'); 2) suggest the most relevant content.

Powered by **[implicit](https://github.com/benfred/implicit)**, **[scikit-learn](http://scikit-learn.org/)** and **[PySpark](https://spark.apache.org/docs/2.4.0/api/python/)**. Distributed computing courtesy of Google Cloud Platform. Data originates
from a leading music streaming platform in China, and was obtained through the [BitTiger Data Science Bootcamp](https://www.bittiger.io/). The project was coded and executed on a 2015 Macbook Pro running macOS 10.13 (High Sierra) + Python 3.7.2.

### Overview of Data Set
- **Day-level** behavioral data: plays, downloads and searches of songs by users
- Date range: 2017/03/30 - 2017/5/12 (6 weeks long)
- For each play event: whether it is paid for, and its duration in seconds
- Limited song metadata available: track name, artist, song length

Since the data does not contain explicit ratings by users on songs, user preferences must be *inferred* from past behavior and summarized as a quantitative *implicit* rating. The main difference between implicit and
explicit ratings is that implicit ones serve better as *confidence values* for predicting whether there will be further activity, rather than indicators of actual preference. Imagine the scenario where users looped a long music track many times:
they *most likely* loved it and would listen to it *again* (hence producing activity) ... Or they simply fell asleep despite not especially
liking the track.

The paper ["Collaborative Filtering for Implicit Feedback Datasets"](https://ieeexplore.ieee.org/document/4781121) discusses
the above considerations in detail, and the algorithm it proposes is utilized in this project through [implicit](https://github.com/benfred/implicit) and [PySpark](https://spark.apache.org/docs/2.4.0/ml-collaborative-filtering.html).

### Summary of Results:
1. Treated churn prediction as a binary classification problem; cross-validated **logistic regression**,
**random forest** and/or **gradient boosting** models. Considering how user activity has dropped
precipitously throughout the 6 weeks (see graphs near top of [notebook B1](https://github.com/openerror/SparkMusicBox/blob/master/src/B1_feature_label_generation_with_spark.ipynb)), retaining users is a matter of
survival for this music streaming site. Fortunately, all three models achieved a **[90% recall
](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)** on test data, which means they
can identify the vast majority of churning users. Then, promotions and discounts can be directed to where they are needed.

2. Built a recommendation system for implicit ratings based on **ALS matrix factorization**.
Cross-validated the factorization model locally; AUC ~0.8 achieved on test data.
Then, implemented scripts for running the same model on Google Cloud.

Please see below for details on project methodology and the purpose of each file. File numbering indicates the order in which each file should be executed.

## 1) Data Preprocessing
See code files starting with letter A under directory `src`.
### `A1_create_data_folders.sh`
Create the working directory structure.

### `A2_download_data.ipynb`
Download the compressed raw data from a remote site. Data segregated by date, for now.

### `A3_unpack_and_clean_files.sh`
Unpack the raw data. Then, collate all play, download and search from all dates into
their respective files.

### `A4_etl_down_sample_by_user.ipynb`
*Not for production systems*. Filtered out bot users, and then downsample to ~11% of human users only.
This is only done so that the project can be completed on a laptop.

## 2) Churn Prediction
See Jupyter notebooks starting with letter B under directory `src`. Churn prediction
can be treated as a binary classification problem, where `churn` == 1 and `not churn` == 0.
In addition, "churning", for each user, is defined as the lack of activity during the last
two weeks, keeping in mind that we have six weeks of data in total.

Thus, some terminology: **label window** refers the last two weeks, while **feature window**
refers to the prior four weeks. Features reflecting user behavior are engineered using data
from the feature window *only*.

### `B1_feature_label_generation_with_spark.ipynb`
Load data onto Spark (local mode), and generated the following features for several time
windows.
1. Number of plays, in the last 1, 3, 7, 14 and 30 days of the feature window
2. Number of downloads, ditto
3. Number of searches, ditto
4. Number of days that elapsed between the last event in the feature window, and the start of the label window

### `B2_train_model_sklearn.ipynb`
Invoked `scikit-learn`'s implementation of logistic regression, random forest and gradient
boosting. Even with just hand-tuned hyperparameters, logistic regression achieved an AUC of
0.86 on the test set, while random forest and gradient boosting managed 0.90. In addition,
all three models achieved similar recall rates, which I argued at the notebook's top to be the most
important metric here.

On the other hand, grid-search cross-validation did not lead to observable improvements in model performance.

### `B3_train_model_spark_ml.ipynb`
Ditto as `B2`, but invoking Spark's machine learning capabilities instead.

## 3) Model-Based Recommendation System
See Jupyter notebooks starting with letter B under directory `src`. For operations on
the cloud, see `sql_song.rec.sql` and `cloud_train.py`; the former creates the necessary
SQL tables, and the latter trains a recommendation system on the SQL data.

### `C1_preprocess.ipynb`
Processes data generated by `A3` into descriptions of interactions between each pair of
users and songs within a given timeframe. Specifically,
- the number of plays and downloads per song, per user;
- whether a user has played the entire song.
Search events are not considered because the data does not associate a search with a specific song.

The timeframe considered is deliberately restricted to the last 30 days where data is available.
After all, recent data should better reflect user preference. In addition, for a production system,
limiting the timeframe considered would conserve storage and computation resources, and possibly
speed up predictions.

From discussions with Youtube users, it seems the video streaming site has adopted a
similar strategy. These users observed that after a 2-week hiatus in usage, they no longer
receive personalized recommendations on the front page of [youtube.com](https://www.youtube.com/); instead
they got whatever that is the currently trending. Such an observation lends support to my
decision of only using the last 30 days of available data to make recommendations.

### `C2_compute_rating.ipynb`
Distill an implicit 'rating' from the user-music interactions discovered in `C1`. It is
hoped that this weighted average of user behavior would reflect a user's preference (or dislike)
for each song. The formula and detailed design considerations are described at the notebook's top.

### `C3_test_recommend.ipynb`
Builds and cross-validates an Alternating Least Squares (ALS) algorithm for making recommendations
based on implicit ratings. This algorithm is described in the paper ["Collaborative Filtering for Implicit Feedback Datasets"](https://ieeexplore.ieee.org/document/4781121)
, and is implemented by [implicit](https://github.com/benfred/implicit) and [Spark](https://spark.apache.org/docs/2.4.0/ml-collaborative-filtering.html).

Since [implicit](https://github.com/benfred/implicit) and [Spark](https://spark.apache.org/docs/2.4.0/ml-collaborative-filtering.html)
utilize the same algorithm, the idea here is to iterate quickly using `implicit` on a downsampled
data set and single machine, find an effective set of hyperparameters, and then deploy
the model "for real" onto the cloud via Spark (see `cloud_train.py`).

While I admit there are risks in tuning a model using a minority subset of data, the risks
are outweighed by the efficiency stemming from programming in a familiar, single-machine environment.
It is the difference between getting something done, or nothing at all.
