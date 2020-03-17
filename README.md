# Machine_learning_supervised_learning

TASK : 

To classify suspected FNA cells to Benign (class 0) or Malignant (class 1) using logistic regression as the classifier. The dataset in use is the Wisconsin Diagnostic Breast Cancer (wdbc.dataset).

Plan of work :

1. Extract features values and Image Ids from the data: Process the original CSV data files
into a Numpy matrix or Pandas Dataframe.
2. Data Partitioning: Partition your data into training, validation and testing data. Randomly
choose 80% of the data for training and the rest for validation and testing.
3. Train using Logistic Regression: Use Gradient Descent for logistic regression to train the
model using a group of hyperparameters.
4. Tune hyper-parameters: Validate the regression performance of your model on the validation
set. Change your hyper-parameters. Try to find what values those hyper-parameters should take
so as to give better performance on the validation set.
5. Test your machine learning scheme on the testing set: After finishing all the above
steps, fix your hyper-parameters and model parameter and test your models performance on the
testing set.

EVALUATION :

Precision, Recall, Accuracy
