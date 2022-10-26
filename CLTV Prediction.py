import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import datetime as dt
import matplotlib as plt

#It is not ethical for me to share this dataset.
df_ = pd.read_csv(r"C:\Users\Sony\PycharmProjects\pythonProject1\CRM Analytics\########")
pd.set_option("display.max_columns",None)
df = df_.copy()
df.head()

def outlier_threshold(dataframe,col):
    q1 = dataframe[col].quantile(0.01)
    q3 = dataframe[col].quantile(0.99)
    ıqr = q3-q1
    low = q1 - 1.5*ıqr
    up = q3 + 1.5*ıqr
    return low,up

def replace_outlier(dataframe,col):
    low,up = outlier_threshold(dataframe,col)
    (dataframe.loc[dataframe[col]> up,col]) = round(up,0)
    (dataframe.loc[dataframe[col]< low,col]) = round(low,0)

df.describe().T

def baskıla(dataframe):
    x = [col for col in dataframe.columns if "total" in col]
    for i in x:
        replace_outlier(dataframe,i)

baskıla(df)

df["total_price"] = df["customer_value_total_ever_offline"]+df["customer_value_total_ever_online"]
df["total_order"] = df["order_num_total_ever_offline"]+df["order_num_total_ever_online"]

df.dtypes

def to_datetime(dataframe):
    x = [col for col in dataframe.columns if "date" in col]
    for i in x:
        dataframe[i] = pd.to_datetime(dataframe[i])
    return dataframe[x].info()

to_datetime(df)

df["last_order_date"].max()
########################################################################################
today_date = dt.datetime(2021,6,1)
cltv = pd.DataFrame()
cltv["customer_id"] = df["master_id"]
cltv["recency_cltv_weekly"] = df["last_order_date"] - df["first_order_date"]
cltv["T_weekly"] = today_date - df["first_order_date"]
cltv["frequency"] = df["total_order"]
cltv["monetary_cltv_avg"] = df["total_price"]

cltv["recency_cltv_weekly"]=cltv["recency_cltv_weekly"].astype("timedelta64[D]")
cltv["T_weekly"] =cltv["T_weekly"].astype("timedelta64[D]")

cltv = cltv[cltv["frequency"]>1]
cltv["monetary_cltv_avg"] = cltv["monetary_cltv_avg"] / cltv["frequency"]
cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"] / 7
cltv["t_weekly"] = cltv["T_weekly"] / 7
########################################################################################
########################################################################################

bgf=BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"],cltv["recency_cltv_weekly"],cltv["t_weekly"])
cltv["exp_sales_3_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(3*4,
                                                        cltv["frequency"],
                                                        cltv["recency_cltv_weekly"],
                                                        cltv["t_weekly"])

cltv["exp_sales_6_month"]=bgf.conditional_expected_number_of_purchases_up_to_time(24,
                                                        cltv["frequency"],
                                                        cltv["recency_cltv_weekly"],
                                                        cltv["t_weekly"])

#################################################################################
#################################################################################

ggf =GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["frequency"],cltv["monetary_cltv_avg"])

ggf.conditional_expected_average_profit(cltv["frequency"],cltv["monetary_cltv_avg"])

#################################################################################
#################################################################################


cltv_final = ggf.customer_lifetime_value(bgf,
                                         cltv["frequency"],
                                         cltv["recency_cltv_weekly"],
                                         cltv["t_weekly"],
                                         cltv["monetary_cltv_avg"],
                                         time=6,
                                         freq="W",
                                         discount_rate=0.01)
cltv_final = cltv_final.reset_index()
cltv_final.index = cltv["customer_id"]
cltv_son = cltv.merge(cltv_final,on="customer_id",how="left")
cltv_son = cltv_son.sort_values(by = "clv",ascending=False)

cltv_son["segment"] = pd.qcut(cltv_son["clv"],4,labels=["D","C","B","A"])

cltv_son.to_csv("cltv_son.csv")