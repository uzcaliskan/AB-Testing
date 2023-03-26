import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 800)
pd.set_option("display.expand_frame_repr", False)  # pd.set_option("display.width", 800) ile aynı işi yapıyor
pd.set_option("display.float_format", lambda x: "%.5f" % x)
###################
## DERECELENDİRMEYE GÖRE SIRALAMAK (Sorting by Rating)
###################

df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/product_sorting.csv")
df.head(20)
df.shape
df.info()
df.isnull().sum()
df.describe().T
df.sort_values("rating",ascending=False).head()
# ÇOK ÖNEMLİ NOT: sadece Dereceye göre / rating e göre sıralama yapmak sosyal kanıt oalrak değerlendirilen ürün satın alma ve ürün sayısını kavramlarını
#kapsamadğındna tek başına fayda vermeycektir.

###################
## YORUM VE SATIN ALMAYA GÖRE SIRALAMAK (Sorting by COMMENT COUNT OR PURCHASE COUNT)
###################
df.sort_values("commment_count",ascending=False).head()
df.sort_values("purchase_count",ascending=False).head()

# YORUM VE SATIN ALMAYA GÖRE DE SIRALAMA YAPILDIĞINDA TEK BAŞLARINA BİR SONUÇ VERMEEYCEKTİR. AZ SATIN ALINMIŞ YÜKSEK RATING Lİ KURSLAR GİBİ
#SATIN ALMASI YA DA YORUM SAYISI YÜKSEK OLUP DÜŞÜK RATINGLI KURSLAR ÜST SIRALAMAYA ÇIKARILMAK İSTENMEZ.
###################
## DERECELENDİRME, YORUM VE SATIN ALMAYA GÖRE SIRALAMAK (Sorting by Rating, COMMENT COUNT AND PURCHASE COUNT):
###################
# NOT: BURADA 3 DEĞİŞKEN DE AYNI CİNSDEN (1-5 ARASI) İFADE EDİLEREK AĞIRLIKLI ORTALAMA YAPILARAK SIRALANMAYA ÇALIŞILIYOR !

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \ #burada ölçek aralığının 1-5 arasında olacağı söyleniyor
    fit(df[['purchase_count']]). \ #burada MinMaxScaler fonksiyonu fit edilerek sonradan ölçeklendirme için kullanılacak minimum ve maximum değer hesaplanıyor
    transform(df[["purchase_count"]]) # burada ise MinMaxScaler fit edildilkten sonra ortaya çıkan sonuç numpy array i olarka transform ediliyor !

df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[['purchase_count']]).transform(df[["purchase_count"]])
df["commment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)).fit(df[['commment_count']]).transform(df[["commment_count"]])

#ÇOK ÇOK ÖNEMLİ NOT - 1: BURADA RFM DEKİ GİB İ SEGMENTLERE AYIRMA İŞLEMİ YAPILMIYOR VE DE YAPILMAMALI. BURADA AMAÇ SEGMENTLERE AYIRMAK DEĞİL,
# MATEMATİKSEL OLARAK MINMAXSCALER İLE TÜM SAYISAL DEĞŞKENLER 1-5 ARALIĞINDA DAĞITMAK
#ÇOK ÇOK ÖNEMLİ NOT - 2: FIT VE TRANSFORM METODLARI PANDAS DATAFRAME KABUL EDİYOR !!! BU YÜZDEN ÇİFT KÖŞELİ PARANTEZ KULLANILDI

(df["commment_count_scaled"] * 32 / 100 + df["purchase_count_scaled"] * 26 / 100 + df["rating"] * 42 / 100) #BURADAKİ % LİK DEĞERLER YORUMA GÖRE DAĞITILDI

def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["commment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

df["weighted_sorting_score"] = weighted_sorting_score(df)
df.sort_values("weighted_sorting_score", ascending=False).head(20)

## DİKKAT: ELİMİZDEKİ BİRDEN FAZLA FAKTÖRÜ ÖLÇEKLENDİRİLİP AYNI SKALAYA İNDİRGEDİKTEDİKTEN SONRA BU
##FAKTÖRLERİNDE EŞİT YADA TARAFIMIZCA VERİLEN AĞIRLIKLANDIRMASIYLA(YÜZDELENDİRME) BUNLARI SIRALIYORUZ.

###################
## BAYESIAN AVERAGE RATING SCORE (BAYES ORTALAMA DERECELENDİRME SKORU):
###################
# SORTING PRODUCTS WITH 5 STAR RATED
# SORTING PRODUCTS ACCORDING TO DISTRIBUTION OF 5 STAR RATING
import math
import scipy.stats as st
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score #BAR

# BAYES ORTALAMA DERECELENDİRME SKORU: PUAN DAĞILIMLARININ ÜZERİNDEN OLASILIKSAL BİR ORTALAMA HESABI YAPAR.

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                        "2_point",
                                        "3_point",
                                        "4_point",
                                        "5_point"]]), axis=1) #burada axis=1 denilerek x in sadece değişkenlerde gezmesi sağlanıyor!

df[df["course_name"].index.isin([5, 1])]
df.sort_values("bar_score",ascending=False)

# Rating Products
# - Average
# - Time-Based Weighted Average
# - User-Based Weighted Average
# - Weighted Rating
# - Bayesian Average Rating Score #olasılıksal bir ortalama değer olup puanları kırpmaktadır!

# Sorting Products
# - Sorting by Rating
# - Sorting by Comment Count or Purchase Count
# - Sorting by Rating, Comment and Purchase (sorting by weighted weighted_sorting_score yeterli olacaktır)
# - Sorting by Bayesian Average Rating Score (Sorting Products with 5 Star Rated)
# - Hybrid Sorting: BAR(Bayesian Average Score) Score + Diğer Faktorler

# Çok Önemli Not: BAR score'da yorum sayısı ve kullanıcı/ satın alan sayısı ihmal edildiğinden göz ardı ediliyor. ' \
#               '5 yıldızıın dağılımını olasılıksal hesaplıyor. bu yüzden kullanılır olamayabilir

 #Hybrid Sorting
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["commment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)

def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40): #wss_w = weighted sorting score
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point",
                                        "2_point",
                                        "3_point",
                                        "4_point",
                                        "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w / 100 + wss_score * wss_w / 100


df["hybrid_sorting_score"] = hybrid_sorting_score(df)
df.sort_values("hybrid_sorting_score", ascending=False)

#hybrid_sorting_score fonksiyonunda bar_score a %60 ağırlık verilmesi potensiyel kursların yakalanması sağlamıştur.
######################################
# IMDB Film Puanları ve Sıralama
######################################
import pandas as pd
import math
import scipy.stats as st
df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/movies_metadata.csv",
                 low_memory=False) # DtypeWarning: Columns (10) have mixed types. Specify dtype option on import or set low_memory=False.
df.head()

df = df[["title", "vote_average", "vote_count"]]
df.head()

# 2015 yılına kadar kullanılmış olan IMDB algoritması:

df.sort_values("vote_average", ascending=False)
df.describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
df[df["vote_count"] > 400].describe([0.10, 0.25, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99]).T
df[df["vote_count"] > 400].sort_values("vote_average", ascending=False)

from sklearn.preprocessing import MinMaxScaler
df["vote_count_scaled"] = MinMaxScaler(feature_range=(1, 10)). \
    fit(df[["vote_count"]]). \
    transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_scaled"]
# NOT: BURADAKİ ÇARPMA İŞLEMİNDEKİ AMAÇ OY SAYISI İLE OY ORTALAMSI ARASINDA BİR İLİŞKİ KURMAKTIR
df.sort_values("average_count_score", ascending=False).head(20)

######################################
# IMDB AĞIRLIKLI DERECELENDİRME (IMDB WEIGHTED RATING)
######################################

# weighted_rating = ((v/(v+M)) * r) + ((M/(v+M)) * C)

# r = vote average - OY ORTALAMASI(PUAN-RATING)
# v = vote count - OY SAYISI
# M = minimum votes required to be listed in the Top 250 - HESABA KATILACAK FİLMLERİN ALMASI GEREKEN MİN OY SAYISI
# C = the mean vote across the whole report (currently 7.0) - TÜM VERİ SETİNİN OY ORTALAMASI

# Film 1:
# r = 8
# M = 500
# v = 1000

# (1000 / (1000+500))*8 = 5.33


# Film 2:
# r = 8
# M = 500
# v = 3000

# (3000 / (3000+500))*8 = 6.85

# (1000 / (1000+500))*9.5

# Film 1:
# r = 8
# M = 500
# v = 1000

# Birinci bölüm:
# (1000 / (1000+500))*8 = 5.33
#
# # İkinci bölüm:
# 500/(1000+500) * 7 = 2.33

# Toplam = 5.33 + 2.33 = 7.66


# Film 2:
# r = 8
# M = 500
# v = 3000

# Birinci bölüm:
# (3000 / (3000+500))*8 = 6.85

# İkinci bölüm:
# 500/(3000+500) * 7 = 1

# Toplam = 7.85

M = 2500
C = df['vote_average'].mean()

# weighted_rating = ((v/(v+M)) * r) + ((M/(v+M)) * C)
# r = vote average - OY ORTALAMASI(PUAN-RATING)
# v = vote count - OY SAYISI
# M = minimum votes required to be listed in the Top 250 - HESABA KATILACAK FİLMLERİN ALMASI GEREKEN MİN OY SAYISI
# C = the mean vote across the whole report (currently 7.0) - TÜM VERİ SETİNİN OY ORTALAMASI
def weighted_rating(r, v, M, C):
    return  ((v / (v + M)) * r + ((M / (v + M)) * C))

weighted_rating(7.4, 12000, 2500, 5.6)
weighted_rating(8.1, 14075, 2500, 5.6)
weighted_rating(8.5, 8358, 2500, 5.6)

df["weighted_rating"] = weighted_rating(df["vote_average"], df["vote_count"], M, C )
df.sort_values("weighted_rating", ascending=False).head(20)
# ÇOK ÖNEMLİ NOT: SIRALAMA KONUSU YORUMA DAYALI OLDUĞUNDAN HER KURUM KENDİ FORMÜLÜNÜ BULARAK KENDİ FORMÜLÜNE GÖRE SIRALAMA YAPABİLİR.
#   BAYESIAN AVERAGE RATING SCORE SADECE BİLİMSEL ANLAMDA, OLASILIK BARINDIRAN VE DE KENDİSİNE GİRİLEN PUAN DEĞERLERNİNİ KIRPARAK AZALTAN BİR BİLİMSEL YAKLAŞIMDIR
#   BAR SCORE HESAPLANIRKEN FORMÜLE GİRİLEN PUANLARA KAÇAR TANE OY VERİLDİĞİ(PUANLARIN OY DAĞILIMLARI) BİLİNMELİDİR !
####################
# Bayesian Average Rating Score
####################
def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score

df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/imdb_ratings.csv")

df = df.iloc[:, 1:]

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis=1)

#ÇOK ÖNEMLİ NOT: BAYESIAN AVERAGE RATING FORMULU KULLANILIRKEN, FORMÜLE SÜTUNİSİMLERİ DATAFRAMEDEKİ SÜTUN SIRALAMASININ TERSİNDE GİRİLMELİ.
# DATAFRAME DEKİ SÜTUNLARIN SIRALAMASI = ["ten",    "nine",  "eight"  , "seven"   , "six"  , "five",   "four"  ,"three"  , "two"   , "one"]
#FORMÜLE GİRİLEN SIRA = x[["one", "two", "three", "four", "five","six", "seven", "eight", "nine", "ten"]])
df.sort_values("bar_score", ascending=False)

# Weighted Average Ratings
# IMDb publishes weighted vote averages rather than raw data averages.
# The simplest way to explain it is that although we accept and consider all votes received by users,
# not all votes have the same impact (or ‘weight’) on the final rating.

# When unusual voting activity is detected,
# an alternate weighting calculation may be applied in order to preserve the reliability of our system.
# To ensure that our rating mechanism remains effective,
# we do not disclose the exact method used to generate the rating.
#
# See also the complete FAQ for IMDb ratings.

# not: bayesian ile verilen puana ağırlık verildğinden potansiyel ürün ya da kursları bulabiliyorduk.