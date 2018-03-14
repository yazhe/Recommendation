# Recommendation
a sample project of keyword recommendation
## Prerequisites
This project requires to install the [jieba](https://pypi.python.org/pypi/jieba/) python package for Chinese word segamentation. It also requires the [sklearn](http://scikit-learn.org/stable/) package for the machine learning (ML) based recommendation function. 
Other packages required include [numpy](http://www.numpy.org/),[scipy](https://www.scipy.org/), and [pandas](https://pandas.pydata.org/) packages.

You can use the pip to install all the required packages:
```
pip install numpy
pip install scipy
pip install pandas
pip install sklearn
pip install jieba
```
## Getting started
This section describs how to use the recommender package to recommend keywords to products. 
### Input datasets
The recommender learn the relevence of products and query keywords from a log event data file. The log event data file must contain records of *product title*, *category*, *query*, *event (Impression/Click)*, *date*. A sample log data file can be found in [sample.csv](data/sample.csv). 

Then, the recommender can provide keyword recommendation for a given product with *product title* and *category*. Some sample test cases can be found in [test.csv](data/test.csv). 
### Load data
You must allow the recommender to read log event data and build relevent indexes before running the recommendation functions.

This process can be done via loading the sample data file directly:
```
from recommender.core import Recommender

rmd = Recommender()
rmd.load_data_from_file("./data/sample.csv")
```
or via loading a dataframe of the sample data:
```
import dataframe as df

data = df.read_csv("./data/sample.csv",sep=',')
rmd.load_data_from_df(data)
```

There are two keyword recommendation functions avaliable:
### IR-based recommendation
The IR-based recommendation function takes 4 inputs:
- **product title**: the title of a product that needs recommendation.
- **category**: the category that the product belong to.
- **top_k**: the number of recommended keywords returned.
- **alpha**: the value in the range (0,1). This value is used to tune the weight of the click rate in the final ranking score of the recommendation. The higher is this value, the function is more likely to recommend keywords with high click rate. 

It outputs a list of *k* recommended keywords
```
product_title = "【現貨】夏季Burberry 巴寶莉 純棉 logo 素T 短T 戰馬 經典 印花 加大尺碼 短袖T恤"
category = "Male Fashion"
top_k = 5
alpha = 0.2
result = rmd.recommend_ir_method(product_title, category, top_k, alpha)
```

### ML-based recommendation
Before running the ML-based recommendation, you must train a classifier using the sample data. The classifer is used to determine for a given product and any keyword, how likely they will result in a click event. 

The train function takes 2 inputs:
- ** data_train_cat**: a data frame that contains only the sample data of a given category
- ** category**: the category of the data that is trained

It output a classifier model that can be used for later recommendation.
```
category = 'Male Fashion'
data_train_cat = data.loc[data['category'] == category]
model = rmd.recommend_ml_train(data_train_cat, category)
```
You can save the model to a file:
```
from sklearn.externals import joblib
joblib.dump(model, 'filename.pkl')
```
Later, load a pre-trained classifier from file:
```
model = joblib.load('filename.pkl') 
```
Then, you can proceed to making recommendations using the ML-based recommendation function. 

The ML-based recommendation function takes 4 inputs:
- **product title**: the title of a product that needs recommendation.
- **category**: the category that the product belong to.
- **top_k**: the number of recommended keywords returned.
- **model**: the pre-trained classifiation model of the given category of products. 

It outputs a list of *k* recommended keywords
```
product_title = "【現貨】夏季Burberry 巴寶莉 純棉 logo 素T 短T 戰馬 經典 印花 加大尺碼 短袖T恤"
category = "Male Fashion"
top_k = 5
result = rmd.recommend_ml_predict(row['product'], cat, top_k, model)
```
## Testing and run
The [test.py] (sourc code/test.py) file provides several detailed exmaples of testing functions of evaluating the performance of the IR-based and ML-based recommendation, and recommendation functions of generating recommendation results for a given test file. 

## Algorithm description
For a detailed description of the algorithms used in the two recommendation function, please refer to the [description.docx](description.docx) file. 
