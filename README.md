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

####Results of Training:

