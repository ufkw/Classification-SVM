# Classification-SVM


## Multi-class classification using least-squares regression on four datasets

#### Process for Choosing Datasets


<p>When choosing datasets for applied machine learning, it's important to practice on high-quality, real-world (non-contrived) datasets. Fortunately, the UCI Machine Learning Repository has a free online database where we could download several high-quality datasets. </p>
<p>For our project, the datasets ‘Car’ and ‘Income’ were chosen originally from the UCI Machine Learning Repository, in addition to the required datasets of ‘Iris’ and ‘Wine’. These two specific datasets were chosen because we wanted to make sure that each set had different numbers of attributes ('Car' having 6 attributes and 'Income' having 14). However, after some experimentation, we found that the dataset 'Income' was too large (having some fifty times more instances) for our model to quickly respond due to the processing power required. This lead us to replace the 'Income' dataset with the dataset 'Ecoli' (with 'Ecoli' having 8 attributes and far less instances, along with real attribute characteristics). It was important for our selection that neither dataset have any missing values (which they did not) as well. We also insured that the datasets ‘Car’ and ‘Ecoli’ had more than 2 classes so that the one vs one and one vs all tests would produce different results. </p>
<p>For the cross validation, we actually just took the entire dataset and split it into 10 folds. We used this process on several lambdas, and then for each fold used the best fit lambda to determine the classification error for that fold. The average classification error calculated was the average of these ten tests and their respective lambdas. By doing this, we could then use the average classification error to find a better lamba, in the code referred to as “optimum_lambda.” </p>
<p>The two datasets we chose are the Car Evaluation and E Coli datasets.  Both sets were chosen to closely match with the traits of the iris and wine datasets.  We insured that the datasets had more than 2 classes so that the one vs one and one vs all tests would produce different results.  We tried to keep the number of instances down to make the machine learning process time manageable.</p>

*The Car Evaluation dataset*
* Attributes: 6; buying, maint, doors, persons, lug_boot, safety
* Classes: 4; unacc, acc, good, v-good

*The Ecoli dataset*
*	Attributes: 8; sequence, mcg, gvh, lip, chg, aac, alm1, alm2
*	Classes: 8; cp, im, pp, imU, om, omL, imL, imS

Out of all the datasets, only E Coli caused MATLAB to warn that the results of the classifier may be inaccurate due to the matrix being close to singular. This warning was not received on the SVM and only on the homegrown, which indicates the sophistication of the SVM.

#### Results of Training:

##### Iris Data

![iris_1](/photos/iris_1.JPG)
![iris_2](/photos/iris_2.JPG)

##### Wine Data

![wine_1](/photos/wine_1.JPG)
![wine_2](/photos/wine_2.JPG)

##### Car Data

![car_1](/photos/car_1.JPG)
![car_2](/photos/car_2.JPG)

##### E Coli Data

![ecoli_1](/photos/ecoli_1.JPG)


## Comparison with multi-class support vector machine in MATLAB (or equivalent)

#### Discussion of one against one (OAO/OVO) versus one against all (OAA/OVA)


The main difference between the one-against-one and one-against-all approaches is the number of classifiers that it takes to learn. 
With the one-against-all approach (the most common used approach), training is done on one classifier per class, having a total of N classifiers. This method is efficient computationally, because only N classifiers are needed. We build one Support Vector Machine per class and train it to distinguish the samples in a single class from the samples in the remaining classes.

For example, for a class I, this method will assume I-labels as positive and the rest as negative. One problem with this approach is that it can lead to imbalanced datasets. This means that a generic Support Vector Machine might not work well using this method. For this reason, it's important that the Support Vector Machines be well tuned when using this approach. It's important to note that the binary classification learners still see imbalanced distributions because of the greater set of negatives, however. 

With the one-against-one approach, separate classifiers are trained for each different pair of labels. We build one Support Vector Machine for each pair of classes. Because of this, the training process is faster. In this approach, we have [(N*(N-1))/2] binary classifiers. For N classes, Support Vector Machines are trained to distinguish the pair samples of one class from the samples of another class. All binary classifiers are tested, then the class with the most votes is promoted at prediction time. Ties are determined by the class with the highest aggregate classification confidence (by summing over the confidence levels of the pairs, computed by the underlying binary classifiers). While this method is more expensive to compute (and slower), it leads to less problems with imbalanced sets. This is especially true of something like kernel algorithms, because they don't scale well with N samples.

#### IRIS Data SVM

##### One-Vs-One

![iris_1v1](/photos/iris_1v1.JPG)

##### One-Vs-All

![iris_1vA](/photos/iris_1vA.JPG)

##### Confusion Matrix for Iris at 50/50 ratio with the SVM (One vs all):

![iris_M](/photos/iris_M.JPG)

#### WINE Data SVM

##### One-Vs-One

![wine_1v1](/photos/wine_1v1.JPG)

##### One-Vs-All

![wine_1vA](/photos/wine_1vA.JPG)

##### Confusion Matrix for Wine at 50/50 ratio with the SVM (one vs all):

![wine_M](/photos/wine_M.JPG)

#### CAR Data SVM

##### One-Vs-One

![car_1v1](/photos/car_1v1.JPG)

##### One-Vs-All

![car_1vA](/photos/car_1vA.JPG)

##### Confusion Matrix for Car data at 50/50 ratio with the SVM (one vs all):

![car_M](/photos/car_M.JPG)

#### E COLI Data SVM

##### One-Vs-One

![ecoli_1v1](/photos/ecoli_1v1.JPG)

##### One-Vs_All

![ecoli_1vA](/photos/ecoli_1vA.JPG)

##### Confusion Matrix for E Coli data at 30/70 ratio with the SVM (one vs all):

![ecoli_M](/photos/ecoli_M.JPG)


#### Results of Training

MATLAB has the capability for using the same seed for the randomization, so the randomization for the training patterns and in the cross-evaluation is consistent.

#### Results of Testing

For a quick comparison, we can evaluate the average classification error from the cross-evaluation for the homegrown and the SVM. With the exception of the E Coli data, the SVM performs better. 

##### Average Classification Error

![average_classification_error](/photos/avg_err.JPG)



