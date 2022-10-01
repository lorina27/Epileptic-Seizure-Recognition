# epileptic-seizure-recognition

The main characteristics of the dataset are:

Attribute information: 

The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds. The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So, we have total 500 individual’s recording and each has 4097 data points for 23.5 seconds.

Data distribution: 

We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time. So now we have 23 x 500 = 11500 pieces of information (row), each information contains 178 data points for 1 second (column), the last column represents the label y {1,2,3,4,5}. The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178.

Classification of data:

Variable y contains the category of the 178-dimensional input vector. Specifically, y is in {1, 2, 3, 4, 5}:
5 - Eyes open; means when they were recording the EEG signal of the brain the patient had their eyes open
4 - Eyes closed; means when they were recording the EEG signal the patient had their eyes closed
3 – Yes, they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
2 - They recorded the EEG from the area where the tumor was located
1 - Recording of seizure activity

Re-organization of data:

All subjects falling in classes 2, 3, 4, and 5 are subjects who did not have epileptic seizure. Only subjects in class 1 have epileptic seizure. Although there are 5 classes, we will perform binary classification, class 1 (Epileptic seizure) against the rest.

Approach:

After we import the dataset from UCI repository, we convert target values to binary because we aim for binary classification of seizure vs. non-seizure. We have an imbalanced data set (80% non-seizure vs. 20% seizure), so we use Receiver Operating Characteristic – Area Under the Curve (ROC-AUC) along with the F1 score on the positive minority class to evaluate the performance. We start by evaluating the data using some non-deep learning methods (Logistic Regression, Naïve Bayes, Random Forest) to see how well it classifies with the data and we follow with a deep learning model to predict seizure activity. Our strategy is to incrementally improve our deep learning model to approach State of the Art (SotA). Because the measurement of the incremental improvements can be small and are slightly variable due to random dataset variations, we use stratified k-fold (k=10) with each change to average the results across many iterations.

Test Environment

All processing and experiments were performed using Google Colab. Google Colab is a free online cloud-based Jupyter notebook environment that allows us to train our machine learning and deep learning models on CPUs, GPUs, and TPUs [11]. Colab was chosen due to the availability of advanced processing hardware (GPU: up to Tesla K80 with 12 GB of GDDR5 VRAM, Intel Xeon Processor with two cores @2.20 GHz and 13 GB RAM & TPU: Cloud TPU with 180 teraflops of computation, Intel Xeon Processor with two cores @2.30 GHz and 13 GB RAM) for machine learning tasks, its ease of sharing between team members, and zero cost. The software utilized was Python 3.6.9, NumPy 1.18.2, Tensorflow 2.2.0-rc3, and SciKit-Learn 0.22.2.post1.

Data Preparation and Evaluation

It has been shown that it is important to normalize data when using it within a machine learning context. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the range of values. As part of our data preparation phase, we normalize the independent variables so that the mean of each column equals zero, while the standard deviation equals one, using the following procedure:

![image](https://user-images.githubusercontent.com/33736760/193426822-53eb46f9-e0fd-45a5-9d23-8f3faec52aa2.png)


Where xi is a data point, x̄ is the sample mean and s is the sample standard deviation. Finally, we divide the data into separate, non-overlapping train, test, and validate segments using sklearn model selection, with care taken to ensure a proportionate representation of the positive target Y variables are included in each segment.

Model Evaluation 

Due to the nature of the problem (binary classification of an unbalanced dataset), we elected to use the Receiver Operating Characteristic – Area Under the Curve (ROC-AUC) along with the per-class F1 score for evaluating the performance of the model. The higher the AUC, the better the model is at distinguishing between patients with epilepsy and patients with no epilepsy. The probabilistic interpretation of ROC-AUC score is that if you randomly choose a positive case and a negative case, the probability that the positive case outranks the negative case according to the classifier is given by the AUC. Mathematically, it is calculated by area under curve of sensitivity (TPR) vs. FPR curve (1 – specificity): 

<img width="201" alt="image" src="https://user-images.githubusercontent.com/33736760/193426916-f0f041e4-769c-4941-828c-0e2b9def96e8.png">

<img width="252" alt="image" src="https://user-images.githubusercontent.com/33736760/193426920-358ec39d-74fd-474c-9964-9408a65f1649.png">

<img width="252" alt="image" src="https://user-images.githubusercontent.com/33736760/193426962-18a8e9b8-762c-4145-951f-8182ac76b605.png">


The reason why we supplement our evaluation metrics with F1 score is because F1 score is more efficient when we have an uneven class distribution. For F1 score to be high, both precision (information about a classifier’s performance with respect to false positives) and recall (information about a classifier’s performance with respect to false negatives) should be high. In our dataset, classes 2-5 are considered as non-seizure so we convert them to 0s, and class 1 is considered a seizure so we convert it to 1s. Since we have a data imbalance between 0-values and 1-values, ROC will average over all possible thresholds so F1 is a better metric to mitigate this problem. This is the situation of accuracy paradox where a simple model may have a high level of accuracy but be too crude to be useful. Precision and recall are better metrics in these situations.

In our approach, we aim to maximize ROC-AUC and F1 score in the positive minority class. With the help of a confusion matrix, we are able to visualize and differentiate between the number of true positives, true negatives, false positives, and true negatives. Confusion matrix (also known as an error matrix) is a performance measurement and it is extremely useful for measuring Recall, Precision, Specificity and Accuracy on a set of test data for which true values are known.

Incremental Improvement
Due to the variability encountered with each change to the deep learning model, we adopt a method of using a Stratified K-fold with K=10 in order to robustly test each change in the model to ensure the observed results are most likely due to substantial predictive improvements in the architecture rather than the train and validation data we happen to encounter during a random iteration. In k-fold cross-validation, the data is divided into k folds. The model is trained on k-1 folds with one fold held back for testing. This process gets repeated to ensure each fold of the dataset gets the chance to be the held back set. Once the process is completed, we can summarize the evaluation metric using the mean or/and the standard deviation. Stratified K-Fold approach is a variation of k-fold cross-validation that returns stratified folds, i.e., each set containing approximately the same ratio of target labels as the complete data. With each iteration of the stratified K-fold, we accumulate the predictions within each iteration and generate the evaluation statistics at the end of all folds, thus giving us a more accurate measure of ROC-AUC due to the larger sample size.

The general procedure is as follows:

Shuffle the dataset randomly.
Split the dataset into k groups
For each unique group:
Take the group as a hold out or test data set
Take the remaining groups as a training data set
Fit a model on the training set and evaluate it on the test set
Retain the evaluation score and discard the model
Summarize the skill of the model using the sample of model evaluation scores

RESULTS

Data Preparation
First, the data is acquired directly from the University of California at Irvine (UCI) repository at http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv. The is in comma-separated value format and consists of 11,500 rows and 179 columns of data.

Next, we import the data for preprocessing using NumPy into two different groups. The first 178 columns of data represent the independent variables (X), while the last column is the dependent variable (Y).  The Y variable is then converted from the range [1,5] into [0,1] in order to represent the data as either seizure activity (Y=1) and non-seizure activity (Y=0) using the rule as defined by:
[insert math graph:  if y=1 then 1 else 0]

We then normalize the data and segment into non-overlapping train, validate, and test segments as described above.

After segmentation, the samples are distributed as follows:

<img width="267" alt="image" src="https://user-images.githubusercontent.com/33736760/193427104-121aad62-0b3a-48ec-8dd2-5c6dbbecae16.png">

Table 1:Distribution of data samples

*Classical Machine Learning Evaluation

In order to evaluate the performance of our proposed method, we first examine some classical machine learning algorithms (as implemented in SciKit-Learn).  These methods were chosen due to their previous academic usage for imbalanced data [18,19,20] and, in the case of logistic regression, our study in the classroom.
Below is a table of the selected algorithms and their performance on the data set

<img width="267" alt="image" src="https://user-images.githubusercontent.com/33736760/193427144-a5462664-13bc-4c41-a0a3-a399a76496a2.png">

Table 2:  Performance of Classical Machine Learning Algorithms

<img width="252" alt="image" src="https://user-images.githubusercontent.com/33736760/193427154-b8fcfb27-3aeb-4b41-9896-26a5ec52f176.png">

Figure 1: Confusion Matrix for the Random Forest Classifier

*Deep Learning Model Development

We built our deep learning model using Keras, which is a part of Tensorflow. Each of our Keras models uses the sigmoid as the last-layer activation function and uses the binary cross-entropy loss function to control and evaluate learning.  The activation and loss functions were chosen because our goal is binary classification. Each fully connected layer used the ReLU activation function.

Because of the imbalanced nature of the data, we intentionally increased our batch size from the default 32 to 64 to ensure each batch had a sufficient chance of containing an adequate quantity of positive samples.  Additionally, each model was trained for 100 epochs and, unless otherwise specified, utilizes the RMSprop optimizer with the 
default values.
Table 3 summarizes the various model configurations that were tested against the training and validation sets. 

<img width="544" alt="image" src="https://user-images.githubusercontent.com/33736760/193427188-9f4e81d3-d282-4167-88c7-b955821d8f0c.png">

Table 3:  Various Model Configuations and Performance on the training vs. validation data sets

*Model Selection and Evaluation on Test Data

Model #10 was selected due to its high overall ROC-AUC and F1 score on the minority class. We then evaluated this model against the previously unseen test data segment. The selected architecture and results are summarized below:

<img width="267" alt="image" src="https://user-images.githubusercontent.com/33736760/193427281-5938656e-b9ca-4062-b101-ec6ab97be21e.png">

Table 4: Performance of the Selected Model on the Test Dataset

<img width="252" alt="image" src="https://user-images.githubusercontent.com/33736760/193427289-07ee41bd-7fd3-4945-82f2-90fe7dba72ed.png">

Figure 3: Confusion Matrix of the Selected Model on the Test Dataset

CONCLUSION
This study shows that it is possible to use artificial neural networks to reliably recognize epileptic seizure activity while minimizing the false negative rate of epileptic seizure detection on an imbalanced dataset.  Compared to the various non-deep learning classifiers, we were able to achieve meet/exceed ROC-AUC performance and notably reduce the false negative rate, but at the expense of the false positive performance (though this trade-off can be adjusted via parameters).  Hence, artificial neural networks can be a valuable tool, alongside expert analysis, for the detection of epileptic seizures. 
