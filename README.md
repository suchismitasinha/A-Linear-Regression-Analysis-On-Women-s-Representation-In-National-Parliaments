# A-Linear-Regression-Analysis-On-Women-s-Representation-In-National-Parliaments
 PRE-PROCESSING DATA FOR ANALYSIS :
The data is almost clean and properly labeled however for
easy prediction and calculation of metrics, we change the null
values into zeroes. This makes it easier for us to perform the
data analysis as the platform can easily identify null values as
zero and make appropriate calculations based on that and the
utilized algorithm.

CHOICE OF ALGORITHM: 
Since the data inside the dataset is labeled, we use one of
the supervised learning algorithms to perform the data analysis
and it is expected that our linear regression model should
perform well on the dataset of such size. 

Through Databricks the dataset called ”Seats.csv” is uploaded
and opened in a notebook. A cluster created for our
project was attached to this notebook. Once the dataset was
launched in the notebook, it was converted into a dataframe as
dataframes are easier to work with. The dataset was visualized
in the notebook using the select SQL query.
Now, to begin with, building our model, the necessary
libraries were imported. The library LinearRegression minimizes
the specified loss function (squaredError and an amalgam
of squared error for reasonably small errors and absolute
error for comparatively large ones), with regularization. The
VectorAssembler combines several columns into a vector column.
The features are standardized by the StandardScaler by
eliminating the mean and scaling to unit variance using statistics
of column summary on the training set samples whereas
Pipeline functions as an estimator. A Pipeline comprises of
a series of stages, each of which is either an Estimator or a
Transformer.
The data loaded into a dataframe was then explored and
using the display command, the data was described. The
features were put into a vector and scaled. The primary step
to building our machine learning pipeline was to convert the
predictor features from DataFrame columns to feature vectors
using the pyspark.ml.feature. VectorAssembler() method. The
vectorizer takes the VectorAssembler(). The years 1990-2017
have been taken as input columns to the vectorizer. The “features”
column represents the output column for our analysis.
The data was split into the training and the test set where only
20% of the data was allotted for the test data and the remaining
for the training data. In the next step, a pipeline was built for
our linear regression model which took the stages of vectorizer
and linear regression learner. With the calling of Pipeline.fit(),
the stages lr and vectorizer were executed in order. Figure 1
shows a schematic diagram of our linear regression pipeline. 
Our model made predictions based on our chosen label,
2018 and features. The new dataframe was created that transformed
the test data of the model to yield our predicted values
and check how perfectly our model fits on our chosen dataset.
After predictions, the RMSE, MSE, MAE and the R-squared
values were calculated to calculate the accuracy of our model.
