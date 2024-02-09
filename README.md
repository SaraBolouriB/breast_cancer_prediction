# Breast Cancer Prediction
## Introduction
<p align="justify">
Breast cancer has become a concerning issue in recent years. In many cases, separation of limbs is the only way to prevent it, if it is diagnosed at the last stage. As a result, a good predictor of this issue can be fruitful in successful diagnosis. The main focus of this program is to perform different <b>machine learning classification algorithms</b> to correctly predict the target class and improve it by checking the effectiveness of the original Wisconsin Breast Cancer dataset (WDBC) for breast cancer diagnosis prediction. After running classifiers on the dataset, the comparison was made among them to find the best-performing algorithm and then effective attributes of the dataset were analyzed to improve performance further. In this program, algorithms, <b>including Naïve Bayes, Support Vector Machine (SVM), Multilayer Perceptron (MLP), J48, and Random Forest</b>, have been used. I have used performance metrics: <b>Accuracy, Kappa statistic, precision, recall, F-measure, MCC, ROC area, PRC area</b>.
</p>

## Result
### Study of the algorithm's performance
<p align="justify">
I first demonstrated the obtained result of proposed classifier methods with different parameters. In this section, 10-fold cross validation is used. By checking all parameters to find the better-performing algorithms, it came to the point that Naive Bayes gives the best result among other algorithms. 
</p>
<div align="center">

|Classification Name|Accuracy|Kappa|Precision|Recal|F-measure| MCC | ROC | PRC |
|---                |  :---: |:---:|  :---:  |:---:|  :---:  |:---:|:---:|:---:|
|    Navie Bayes    |  0.970 |0.934|  0.934  |0.985|  0.958  |0.936|0.937|0.925|
|        J48        |  0.928 |0.838|  0.930  |0.929|  0.929  |0.839|0.975|0.965|
|   Random Forest   |  0.964 |0.926|  0.934  |0.964|  0.946  |0.924|0.959|0.909|
|        SVM        |  0.955 |0.901|  0.946  |0.928|  0.935  |0.903|0.949|0.903|
|        MLP        |  0.964 |0.921|  0.961  |0.939|  0.949  |0.923|0.959|0.923|

![image](https://github.com/SaraBolouriB/breast_cancer_prediction/assets/45979215/638972c0-3942-4e2a-a234-179206042a41)
</div>

### Performance improvement study based on the dataset analyzing and modifying
<p align="justify">
As the Naïve Bayes classifier worked best among our proposed classifiers, I tried to optimize the result further. I tried to find the effectiveness of each feature and their effects on the performance. After removing the feature that has less impact on the dataset and negative effects on accuracy, I get better accuracy with a better result, shown below.
<p/>
<div align="center">
  
|Test Mode                     |Accuracy|Kappa|Precision|Recal|F-measure| MCC | ROC | PRC |
|---                           |  :---: |:---: |  :---:  |:---:|  :---:  |:---:|:---:|:---:|
| 5-Fold cross validation      |97.4212 |0.9435|  0.975  |0.974|  0.974  |0.944|0.993|0.993|
|10-Fold cross validation      |97.4212 |0.9435|0.975    |0.974|0.974    |0.944|0.993|0.993|
|15-Fold cross validation      |97.4212 |0.9435|0.975    |0.974|0.974    |0.944|0.993|0.993|
|Split 66.6% train, remain test|96.9957 |0.935 |0.971    |0.970|0.970    |0.936|0.993|0.993|
|Split 85.5% train, remain test|99.0099 |0.9788|0.990    |0.990|0.990    |0.979|0.997|0.997|

![image](https://github.com/SaraBolouriB/breast_cancer_prediction/assets/45979215/158c48b6-ce17-49d7-a0d0-c16096b11aa4)
</div>

## How to Run
To run the script and see the result, follow the below steps:
1. pip install -r requirments.txt
2. cd ./code
3. python main.py

## How code works
The main method of the program has been shown in the image below. The code will be explained in the following:
<div align="center">
  
![image](https://github.com/SaraBolouriB/breast_cancer_prediction/assets/45979215/fc5657a9-9e2c-4106-b063-b76ee10950db)
</div>

<p align="justify">

1. First, the data is read after loading the database.
2. Second, the preprocessing gets started. for preprocessing, the following steps have been done:
   - Finding the missing values, which have been set as '?', and replacing them with the digit "1".
   - Calculating the maximum, minimum, mode, median, and average for each attribute in the database.
4. Third, all mentioned classification algorithms have been implemented on the preprocessed database. In this step, all the performance metrics are calculated for each classification algorithm.
5. Fourth, for the best performance result, Naive Bayes, the k-fold validation has been done.
6. Finally, to improve the result, each attribute in the database is dropped, and then calculate the performance again to find out which attribute has the most negative effect on the accuracy.
</p>
