from flask import Flask, render_template, url_for, request
import pandas as pd  
import os
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#%matplotlib inline
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import confusion_matrix
#import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

app=Flask(__name__)


    #return "hi"

@app.route('/')
def home():
    return render_template('home.html')


def ValuePredictor(to_predict_list):
    #to_predict = np.array(to_predict_list).reshape(1,12)
    # Set random seed
    np.random.seed(0)


    data = pd.read_csv("data.csv",names=['sale_code','open_day','days_open','has_theme','activity','n_products','close_day', 'ship_ups','payment_m','by_omg','fulfillment','open_time','close_time','express','fundraising','first_t_date','last_t_date','dealer','mandatory','state','buying_group','trivest','n_reps','size','store_365','user_12','since_year','orderaverage','ordercount','customer_count','order_total','net_unit','count_category','avg_price_base','count_brand','product_price','category'],index_col=False)
    data= data.drop(0, axis=0)
    


#create open days field
    data['first_t_date'] = pd.to_datetime(data['first_t_date'])
    data['last_t_date'] = pd.to_datetime(data['last_t_date'])
    data['open_days_real']=data['last_t_date']-data['first_t_date']

#Group by payment and mandatory

    data = data.groupby('sale_code').agg({'product_price':'first','n_reps':'first','buying_group':'first','trivest':'first','size':'first','store_365':'first','user_12':'first','since_year':'first','open_day':'first','days_open':'first','category':'first','has_theme':'first', 'n_products':'first','close_day':'first', 'ship_ups':'first','by_omg':'first','fulfillment':'first','open_time':'first','close_time':'first','express':'first','fundraising':'first','first_t_date':'first','last_t_date':'first','orderaverage':'first','ordercount':'first','customer_count':'first','order_total':'first','state':'first','count_brand':'first','count_category':'first','avg_price_base':'first','net_unit':'first','dealer':'first','open_days_real':'first',    
                             'payment_m': ', '.join,'mandatory': ', '.join }).reset_index()






#create has mandatory

    data["days_open"] = data.days_open.astype(float)
    data["orderaverage"] = data.orderaverage.astype(float)
    data['ordercount']=data.ordercount.astype(float)
    data['customer_count']=data.customer_count.astype(float)
    data['order_total']=data.order_total.astype(float)
    data['n_products']=data.n_products.astype(float)
    data['open_days_real']=data.open_days_real.dt.days
    #data['open_days_real']=data.open_days_real.astype(float)
    data['avg_price_base']=data.avg_price_base.astype(float)
    data['count_category']=data.count_category.astype(float)
    data['count_brand']=data.count_brand.astype(float)
    data["n_reps"] = data.n_reps.astype(float)
    data["store_365"] = data.store_365.astype(float)
    data['user_12']=data.user_12.astype(float)
    data['product_price']=data.product_price.astype(float)
    data['since_year']=data.since_year.astype(float)

    data['totalpp']=data['order_total']/data['customer_count']
    data['avg_price']=data['product_price']/data['customer_count']
    data['years']=2019-data['since_year']
    data['hasmandatory']= data['mandatory'].str.contains("Yes")

    #check rows with missing values
    data[pd.isnull(data).any(axis=1)]

    #drop na 🐱‍👤🐱‍🚀

    data=data.dropna(subset=['n_products'])

    #exclude 0 open days real
    data=data[data.open_days_real>0]

    data_weird=data.loc[data['totalpp'] >1000]
    data_weird[['sale_code','totalpp']]
    data = data[data.totalpp <=1000 ]

    data_weird_0=data.loc[data['totalpp'] ==0]

    data_weird_0['totalpp'].hist()

    #data_weird_0[['sale_code','totalpp']]
    data = data[data.totalpp !=0 ]

    data = data[data.totalpp >5 ]

    data.loc[data.totalpp < 100, 'group_2'] = "<100"
    data.loc[data.totalpp>=100, 'group_2'] = ">=100"

    data['hascash'] = data['payment_m'].str.contains("Cash")
    data['hascard'] = data['payment_m'].str.contains("Card")
    data['hasspecial'] = data['payment_m'].str.contains("Speical")

    data['hasshortsleeve'] = data['category'].str.contains("Short")
    data['haslongsleeve'] = data['category'].str.contains("Long")
    data['hasouterwear'] = data['category'].str.contains("Outerwear")

    #below are the more distinguished ones
    #Sport Specific Wearables
    data['haswearables'] = data['category'].str.contains("Wearables")
    #Wearable Accessories
    data['hasaccessories'] = data['category'].str.contains("Accessories")
    #bags
    data['hasbags'] = data['category'].str.contains("Bags")
    #Pants
    data['haspants'] = data['category'].str.contains("Pants")
    #shows
    data['hasshoes'] = data['category'].str.contains("Shoes")
    #specific in total
    data['hasspecific']=data['category'].str.contains('Specific')


    X_2=data[['avg_price','open_day','hasmandatory','days_open','has_theme','haspants','hasbags','hasshoes','hasaccessories','n_products','close_day','count_category','count_brand','by_omg','open_time','close_time','fundraising']]

    Y_2=data['group_2']
    #factorize group label
    Y_2=pd.factorize(data['group_2'])[0]

    category_col =['open_day','has_theme','haspants','hasbags','hasshoes','hasaccessories','close_day','by_omg','open_time','close_time','fundraising'] 
    labelEncoder = preprocessing.LabelEncoder()

    # creating a map of all the numerical values of each categorical labels.
    mapping_dict={}
    for col in category_col:
        X_2[col] = labelEncoder.fit_transform(X_2[col])
        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
        mapping_dict[col]=le_name_mapping
    
    print(mapping_dict)

    X_train, X_test, y_train, y_test = train_test_split(X_2, Y_2, test_size=0.3, random_state=8)

    
    #X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, Y_2, test_size=0.3, random_state=8)

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(n_jobs=3,random_state=0,min_samples_leaf=50,max_depth=20,n_estimators=15,max_leaf_nodes=30)

    clf.fit(X_train, y_train)

    #get accuracy 
    y_pred=clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #precision
    pred_real=pd.DataFrame({'pred': y_pred, 'testlabel': y_test})

    pred_real

    #extract optimal stores
    op1=pred_real[pred_real['pred']==1]
    #extract optimal stores that are classified as optimal stores
    optimal=op1[op1['testlabel']==1]

    precision=len(optimal)/len(op1)
    print(precision)

    #recall

    #extract pred results that are optimal stores
    op2=pred_real[pred_real['testlabel']==1]
    #extract optimal stores that are classified as optimal stores
    optimal2=op2[op2['pred']==1]

    recall=len(optimal2)/len(op2)
    print(recall)


    to_predict_array = np.asarray(to_predict_list)
    to_predict_array=to_predict_array.reshape(1, -1)

    result = clf.predict(to_predict_array)

    return result[0]

@app.route('/result', methods=['GET', 'POST'])
def result():
    avg_price=request.form['avg_price']
    has_theme=request.form['has_theme']
    has_bags=request.form['has_bags']
    has_pants=request.form['has_pants']
    has_shoes=request.form['has_shoes']
    has_accessories=request.form['has_accessories']
    by_omg=request.form['by_omg']
    
    has_fundraising=request.form['has_fundraising']
    open_day=request.form['open_day']
    close_day=request.form['close_day']
    open_time=request.form['open_time']
    close_time=request.form['close_time']
    days_open=request.form['days_open']
    n_brands=request.form['n_brands']
    n_categories=request.form['n_categories']
    n_products=request.form['n_products']
    has_mandatory=request.form['has_mandatory']

    # #put everything into int
    avg_price=int(avg_price)
    has_theme=int(has_theme)
    has_bags=int(has_bags)
    has_pants=int(has_pants)
    has_shoes=int(has_shoes)
    has_accessories=int(has_accessories)
    by_omg=int(by_omg)
    
    has_fundraising=int(has_fundraising)
    open_day=int(open_day)
    close_day=int(close_day)
    open_time=int(open_time)
    close_time=int(close_time)
    days_open=int(days_open)
    n_brands=int(n_brands)
    n_categories=int(n_categories)
    n_products=int(n_products)
    has_mandatory=int(has_mandatory)


    to_predict_list = [avg_price,open_day,has_mandatory,days_open,has_theme,has_pants,has_bags,has_shoes,has_accessories,n_products,close_day,n_categories,n_brands,by_omg,open_time,close_time,has_fundraising]
 
    print(to_predict_list)
    #to_predict_list = list(to_predict_list.values())
    #to_predict_list = list(map(int, to_predict_list))
    result = ValuePredictor(to_predict_list)
    print(result)

    # if int(result) == 1:
    #     prediction = 'Average Order Volume >= $100'
    # else:
    #     prediction = 'Average Order Volume <$100'

    return render_template("result.html", prediction=result,open_day=open_day,avg_price=avg_price,n_products=n_products,days_open=days_open,open_time=open_time,n_brands=n_brands, has_accessories=has_accessories,has_bags=has_bags,has_pants=has_pants,has_shoes=has_shoes,has_theme=has_theme)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
