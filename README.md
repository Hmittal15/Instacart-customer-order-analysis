# **Instacart Customer Order Analysis**
---------------------------------------

The objective of this project is to predict which products a user may want to buy in their next order. Instacart open sourced their transactional data of over 3 million orders, from more than 200,000 Instacart users. We shall use this anonymized data on customer orders over time to predict which previously purchased products will be in a user’s next order. The predictions depend on historical data, leading up to the most recent transactions. Meanwhile, also analyzing the customer data to gain some useful insights about the purchasing pattern, which will be powerful in determining future business operations. With the integration of machine learning algorithms and data analysis platforms, we can impart retailers with robust predictive analytics capabilities, enabling them to stock their stores with the right products at the right time. The algorithms allow the retailers to detect patterns in the various operations and processes of the supply chain. Some additional marketing strategies that retailers might come up with, may include:
* Design better Product Catalog
*	Cross marketing on online stores
*	Roll-out customized emails with add-on sales, etc.

### Description of Dataset:-
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

### Future Work:-
1.	We can implement a solution to this problem using Deep Learning in a more efficient form.
2.	We can extend this solution to provide even more recommendations, such as for each product we can suggest an item which was most frequently purchased with it.

### References:-
*	https://stackoverflow.com/questions/62581004/how-to-set-class-weights-in-decisiontreeclassifier-for-multi-class-setting
*	https://github.com/shubhamscifi/Instacart-Market-Basket-Analysis
*	https://github.com/archd3sai/Instacart-Market-Basket-Analysis
*	https://asagar60.medium.com/instacart-market-basket-analysis-part-1-introduction-eda-b08fd8250502
