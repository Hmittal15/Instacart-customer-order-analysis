import pandas as pd
import pickle
from datetime import datetime

def get_best_prediction(items = None, preds = None, pNone=None, showThreshold = False):
    rec_prod=[]
    for i in range(len(items)):
        if preds[i]>0.3:
            rec_prod.append(items[i])
    return rec_prod

def get_recommendations(X=None):
    start_time = datetime.now()

    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    today = int(dt_string.split("/")[0])

    # get data from user end
    user_id = int(X['user_id'])  # user_id
    order_hour_of_day = int(dt_string.split(" ")[1].split(":")[0])  # current date
    order_dow = datetime.today().weekday()  # current day of week

    ulp = pd.read_pickle("user_last_purchase.pkl")
    if user_id not in ulp['user_id'].values:
        # get top 5 products based on hour of day and day of week
        top = pd.read_pickle('top10_products.pkl')
        top_products = top[(top['order_dow'] == order_dow) & (top['order_hour_of_day'] == order_hour_of_day)][
            'product_name'].values.tolist()
        top_products = {i: value for i, value in enumerate(top_products)}
        # paths = get_image_paths(top5_products)
        predictions = {}
        predictions['top'] = top_products

        del ulp, top, now, today, dt_string, order_dow, order_hour_of_day

        end_time = datetime.now()
        difference = end_time - start_time
        # print("Total Time : {} seconds".format(difference))
        time = "{}".format(difference)

        return predictions, time

    user_last_order_date = ulp[ulp['user_id'] == user_id]['date'].values.tolist()[0]

    days_since_prior_order = today - int(user_last_order_date.split('-')[-1])
    del ulp, now, today, dt_string, user_last_order_date
    # featurization

    hour_rate = pd.read_pickle("hour_reorder_rate.pkl")
    day_rate = pd.read_pickle("day_reorder_rate.pkl")
    p_days_rate = pd.read_pickle("p_days_since_prior_order_reorder_rate.pkl")
    u_days_rate = pd.read_pickle("u_days_since_prior_order_reorder_rate.pkl")
    up_days_rate = pd.read_pickle("days_since_prior_reorder_rate.pkl")

    merged_up_features = pd.read_pickle("Updated_final_dataset.pkl")

    featurized_data = merged_up_features[merged_up_features['user_id'] == user_id]

    hour_r = hour_rate[hour_rate['order_hour_of_day'] == order_hour_of_day]
    day_r = day_rate[day_rate['order_dow'] == order_dow]
    p_days = p_days_rate[p_days_rate['days_since_prior_order'] == days_since_prior_order]
    u_days = u_days_rate[(u_days_rate['user_id'] == user_id) & (u_days_rate['days_since_prior_order'] == days_since_prior_order)]

    if p_days.empty:
        # handle
        p_days = pd.DataFrame(columns=p_days.columns)
        products_x = pd.read_pickle('product_mappings.pkl')
        p_days['product_id'] = products_x['product_id']
        p_days['days_since_prior_order'] = days_since_prior_order
        p_days['p_days_since_prior_order_reorder_rate'] = 0.0

    if u_days.empty:
        # handle
        u_days = pd.DataFrame(columns=u_days.columns)
        df2 = {'user_id': user_id, 'days_since_prior_order': days_since_prior_order,
               'u_days_since_prior_order_reorder_rate': 0}
        u_days = u_days.append(df2, ignore_index=True)
        del df2

    up_days = up_days_rate[(up_days_rate['user_id'] == user_id) & (up_days_rate['days_since_prior_order'] == days_since_prior_order)]

    if up_days.empty:
        # handle
        up_days = pd.DataFrame(columns=up_days_rate.columns)
        products_x = pd.read_pickle('product_mappings.pkl')
        up_days['product_id'] = products_x['product_id']
        up_days['user_id'] = user_id
        up_days['days_since_prior_order'] = days_since_prior_order
        up_days['days_since_prior_reorder_rate'] = 0
        del products_x

    del merged_up_features, hour_rate, day_rate, p_days_rate, u_days_rate, up_days_rate

    featurized_data = pd.merge(featurized_data, up_days, on=['user_id', 'product_id'])

    featurized_data = pd.merge(featurized_data, hour_r, on='product_id')
    featurized_data = pd.merge(featurized_data, day_r, on='product_id')
    featurized_data.drop(['days_since_prior_reorder_rate_x', 'order_dow_x','days_since_prior_order_x','order_hour_of_day_x','hour_reorder_rate_x','day_reorder_rate_x'], axis = 1, inplace = True)
    featurized_data = featurized_data.rename(columns={'days_since_prior_reorder_rate_y':'days_since_prior_reorder_rate', 'order_dow_y':'order_dow', 'days_since_prior_order_y':'days_since_prior_order', 'order_hour_of_day_y':'order_hour_of_day', 'hour_reorder_rate_y':'hour_reorder_rate', 'day_reorder_rate_y':'day_reorder_rate'})
    featurized_data = pd.merge(featurized_data, p_days, on=['product_id', 'days_since_prior_order'])
    featurized_data = pd.merge(featurized_data, u_days, on=['user_id', 'days_since_prior_order'])
    featurized_data.drop(['p_days_since_prior_order_reorder_rate_x','u_days_since_prior_order_reorder_rate_x','reordered'], axis=1, inplace=True)
    featurized_data = featurized_data.rename(columns={'p_days_since_prior_order_reorder_rate_y':'p_days_since_prior_order_reorder_rate', 'u_days_since_prior_order_reorder_rate_y':'u_days_since_prior_order_reorder_rate'})
    featurized_data = featurized_data[['user_id','product_id','total_orders_by_user','user_reorder_rate','add_to_cart_by_user_mean','days_since_prior_order_avg','product_last_bought_order','is_reorder_3','is_reorder_2','is_reorder_1','order_number','order_dow','order_hour_of_day','days_since_prior_order','prod_add_to_cart_order_mean','prod_order_total','prod_reorder_rate','is_organic','aisle_add_to_cart_order_mean','aisle_order_total','aisle_reorder_rate','aisle_0','aisle_1','aisle_2','aisle_3','aisle_4','aisle_5','aisle_6','aisle_7','dept_add_to_cart_order_mean','dept_unique_users','department_0','department_1','department_2','department_3','department_4','dow_mean','dow_std','doh_mean','doh_std','days_since_prior_order_mean','days_since_prior_order_std','products_by_user','reorder_rate_by_user','order_size_avg','orders_3','orders_2','orders_1','reorder_3','reorder_2','reorder_1','hour_reorder_rate','day_reorder_rate','p_days_since_prior_order_reorder_rate','u_days_since_prior_order_reorder_rate','days_since_prior_reorder_rate']]

    del up_days, u_days, p_days, day_r, hour_r

    if featurized_data.empty :
        # get top 5 products based on hour of day and day of week
        top = pd.read_pickle('top10_products.pkl')
        top_products = top[(top['order_dow'] == order_dow) & (top['order_hour_of_day'] == order_hour_of_day)][
            'product_name'].values.tolist()
        top_products = {i: value for i, value in enumerate(top_products)}
        # paths = get_image_paths(top5_products)
        predictions = {}
        predictions['top'] = top_products

        del top, order_dow, order_hour_of_day

        end_time = datetime.now()
        difference = end_time - start_time
        # print("Total Time : {} seconds".format(difference))
        time = "{}".format(difference)

        return predictions, time

    else:
        # model
        with open("random_forest_model.pkl", "rb") as f:
            model = pickle.load(f)
        data = featurized_data.drop(['user_id', 'product_id'], axis=1)
        ypred = model.predict_proba(data)
        ypred = ypred[:, -1]  # get probabilities of class 1
        del data, model

        recommended_products = get_best_prediction(featurized_data['product_id'].tolist(), ypred.tolist(), None)
        products_x = pd.read_pickle('product_mappings.pkl')
        recommended_products = products_x.loc[products_x['product_id'].isin(recommended_products)][
            'product_name'].values.tolist()
        recommended_products = {i: value for i, value in enumerate(recommended_products)}

        predictions = {}
        predictions['recommend'] = recommended_products

        end_time = datetime.now()
        difference = end_time - start_time
        # print("Total Time : {} seconds".format(difference))
        time = "{}".format(difference)

        del featurized_data, products_x
        return predictions, time