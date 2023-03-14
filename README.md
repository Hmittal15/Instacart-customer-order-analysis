# **Instacart Customer Order Analysis**
---------------------------------------

The objective of this project is to predict which products a user may want to buy in their next order. Instacart open sourced their transactional data of over 3 million orders, from more than 200,000 Instacart users. We shall use this anonymized data on customer orders over time to predict which previously purchased products will be in a user’s next order. The predictions depend on historical data, leading up to the most recent transactions. Meanwhile, also analyzing the customer data to gain some useful insights about the purchasing pattern, which will be powerful in determining future business operations.

```
📦 Instacart-customer-order-analysis
├─ .gitignore
├─ Additional features.ipynb
├─ Decision Tree model.ipynb
├─ Demo.gif
├─ Deployment
│  ├─ Derived_data
│  │  ├─ day_reorder_rate.pkl
│  │  ├─ hour_reorder_rate.pkl
│  │  ├─ p_days_since_prior_order_reorder_rate.pkl
│  │  ├─ product_mappings.pkl
│  │  ├─ top10_products.pkl
│  │  ├─ u_days_since_prior_order_reorder_rate.pkl
│  │  └─ user_last_purchase.pkl
│  ├─ Dockerfile
│  ├─ app.py
│  ├─ get_prediction.py
│  ├─ index.html
│  ├─ new_user_recommendation.html
│  ├─ predict.html
│  ├─ random_forest_model.pkl
│  └─ requirements.txt
├─ Feature_Engineering.ipynb
├─ Instacart_customer_order_analysis.mp4
├─ Logistic Regression model.ipynb
├─ Project_EDA.ipynb
├─ README.md
├─ Random Forest model.ipynb
├─ Report.docx
└─ XGBoost model.ipynb
```

### Description of Dataset:- (https://www.kaggle.com/competitions/instacart-market-basket-analysis/data)
The data contains anonymized sample of over 3 million orders from more than 200,000 users. For each user, Instacart provided between 4 and 100 of their orders, along with the sequence in which products were placed in the cart. There are a total of 6 files with a total of 207 Megabytes of data. The entire dataset can be broadly categorized into the following:
1.	Prior data: Order history of every user. This data contains nearly 3-100 past orders per user (~3.2m orders)
2.	Train data: Current order data of every user. This data contains only 1 order per user (~131k orders)
3.	Test data: Future order data of every user . This data will not contain any product information. Essentially, there are the orders for which we have to predict the reorder products for users. (75k orders)

### Methodology:-
Since, we have to predict multiple products for a given order, this might look like a multilabel classification task. There are 49688 products, and total product recommendations could be anywhere from None to N. Therefore, we will restructure this problem into a binary classification problem, where we will predict the probability of an item being reordered by a user. We shall classify each item into 2 categories for any particular user, i.e., if that product will be reordered or not.

To build a machine learning model, we need to extract additional features from orders data to understand user's purchase pattern and popularity of all the products. We shall extract following features from the user's transactional data.
*	Product Level Features
*	Aisle and Department Level Features
*	User Level features
*	User-product Level Features
Generally, we can say if Probability (item reorder) > 0.5 -> Class 1 (reorder in this case) else Class 0. So, we can select those products which belong to class 1 ( i.e., P(X) > 0.5) and recommend them to user. But this threshold 0.5 can be changed in order to improve the performance of the model.

I followed below processes to achieve our goal of predicting a reorder in an efficient manner:-
1.	Data cleaning and pre-processing
2.	Exploratory Data Analysis
3.	Feature engineering
4.	ML algorithms

I have used below machine learning models to solve the classification problem:
1.	Logistic regression
2.	Decision tree classifier
3.	Random Forest classifier
4.	XGBoost classifier


Below are the cumulative performance metrics scores for all the employed ML models:-
</br>
<img width="356" alt="image" align="center" src="https://user-images.githubusercontent.com/108916132/214218640-7adb0c6b-c795-4da0-93d2-38315eb90cf0.png">

Careful and precise analysis was performed over all the aspects of these models. I compared all the performance metrics scores of these models and analyzed the significance of each. Post introspection of all these models, I can narrow down to ‘Random Forest classifier’ as the best performing model for our use-case. Precision and recall scores in predicting both, 0 and 1 are higher for Random Forest algorithm as compared to other models. This means that model can efficiently predict true positive values of 0 and 1. Consequently, the cumulative F1-score for the model is highest. So, I can confidently quote that this model will generate the most efficient predictions in determining whether a product will get reordered by a user in their future order.

### Deployment:-
1. Designed an API using Flask framework to expose the Random Forest model funtionalities.
2. Designed front-end web-application using HTML/CSS to interact with the Flask API endpoints.
3. Containerized the application using Docker container, allowing for easier sharing and scalability. Published the Docker image on DockerHub, making it easily accessible to others over the internet (https://hub.docker.com/r/mittal15/instacart/tags).
4. Deployed the application on a Google Cloud Platform (GCP) VM instance, utilizing top-tier cloud computing infrastructure to provide fast and reliable hosting.

### Application demo:-
![Demo](https://user-images.githubusercontent.com/108916132/225123837-2c155764-73bb-442c-addc-a51cdc54a6b8.gif)

### Link to full explanatory video:-
[![Video thumbnail](![Screenshot (88)](https://user-images.githubusercontent.com/108916132/225125044-ff1d3a60-9b4c-4d29-89b1-cc2fc4bd180d.png)
)](https://github.com/Hmittal15/Instacart-customer-order-analysis/raw/main/Instacart_customer_order_analysis.mp4 "Download or view the video")

### Future Work:-
1.	We can implement a solution to this problem using Deep Learning in a more efficient form.
2.	We can extend this solution to provide even more recommendations, such as for each product we can suggest an item which was most frequently purchased with it.


## You can find me on <a href="http://www.linkedin.com/in/harshit-mittal-52b292131"> <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/LinkedIn_logo_initials.png/768px-LinkedIn_logo_initials.png" width="17" height="17" /></a>
