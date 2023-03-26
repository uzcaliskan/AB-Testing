##BU BÖLÜMDE AMAÇ ÜRÜNLERİN PUANLARININ NASIL HESAPLANMASI GEREKTİĞİNE
# DAİR BİR FİKİR VERMEYE ÇALIŞMAKTIR
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 800)
pd.set_option("display.expand_frame_repr", False)  # pd.set_option("display.width", 800) ile aynı işi yapıyor
pd.set_option("display.float_format", lambda x: "%.5f" % x)
df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/course_reviews.csv")
df.head()
df.shape
df.info()
df["Rating"].value_counts()
df["Rating"].mean()
df["Questions Asked"].value_counts()
df.groupby("Questions Asked").agg({"Questions Asked": "count", "Rating": "mean"})

# Puan Zamanlarına göre ağırlıklı ortalama ( Time Based Weighted Average)
# Buradaki amaç son zamanlardaki trendlere göre ürün puanının hesaplanarak son zamanlardaki trendi yakalamaktır.

df.info()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # timestamp değişkenini zamana çevirdik
# alt2:
df["Timestamp"] = df["Timestamp"].apply(pd.to_datetime)  # timestamp değişkenini zamana çevirdik
import datetime as dt

# current_date = df["Timestamp"].max() + dt.timedelta(2)
current_date = df["Timestamp"].max() + dt.timedelta(5)  # videoda 10 şubat 2021 00:00:00 'a  göre hesap yapılıyor


df["days"] = (current_date - df["Timestamp"]).dt.days

df[df["days"] <= 30].count()  # son 30 gündeki yorumlar
# df[df["days"] <= 30]["Rating"].mean()
df.loc[df["days"] <= 30, "Rating"].mean()  # son 30 gün ortalması

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()  # 3-1 aylar ası ortalama

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()  # 3-6 ay arası ortalama

df.loc[df["days"] >= 180, "Rating"].mean()

df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
df.loc[df["days"] >= 180, "Rating"].mean() * 22 / 100


def time_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           df.loc[df["days"] >= 180, "Rating"].mean() * w4 / 100

time_weighted_average(df)

# User-Based (User-Quality) Weighted Average - Kullanıcı Temelli Ağırlıklı Ortalama
# Burada örnek vermek gerekirse, IMDB'ye yeni üye olup 1 film için oy kullananın ortaalamya etkisi yıllardır
# IMDB'ye üye olup oy kullanan kullanıcının ortalamaya olan etkisinden daha azdır. Amaç Sosyal İspat meselesini en
# doğru şekilde kullanıcıya vermektir.


def user_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return df.loc[df["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           df.loc[(df["Progress"] > 10 ) & (df["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           df.loc[df["Progress"] >= 75, "Rating"].mean() * w4 / 100
user_weighted_average(df)

# Weighted Rating
def course_wieghted_rating(dataframe, time_w=50, user_w=50):
    return time_weighted_average(dataframe) * time_w / 100 + user_weighted_average(dataframe) * user_w / 100

course_wieghted_rating(df, time_w=40,user_w=60)