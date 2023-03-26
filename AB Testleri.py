import itertools
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.expand_frame_repr", False)
############################
# Sampling (Örnekleme)
############################
# NOT: ÖRNEKLEME TEORİSİ; BELİRLİ BİR POPULASYONDAN SEÇİLEN, O POPULASYONUN ÖZELLİKLERİNİ İYİ TEMSİL EDEN BİR ALT KÜMENİN
# ÖZELLİKLERİNİN GENELLENEBİLECEĞİNİ SÖYLER ! YAPILAN GENELLEMELER BİR HATA PAYI BARINDIRIR !


populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()
np.random.seed(115)
# The numpy random seed is a numerical value that generates a new set or repeats pseudo-random numbers.
# The value in the numpy random seed saves the state of randomness. If we call the seed function using value
# 1 multiple times, the computer displays the same random numbers
# burada seed ile aynı random değerlerin görünmesi sağlanıyor
orneklem = np.random.choice(a=populasyon, size=100)
# choice : Generates a random sample from a given 1-D array

orneklem.mean()
np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10

# GENELLİKLE ANA KİTLER ÜZERİNDEN BRİ ÖRNEKLEM ÇEKİLEREK ÇALIŞMALAR YAPILMAKTADIR ! GERÇEKTE DE HER ZAMAN BİR GENELLEME
# TAHMİN VERİ BİLİMİNDE BULUNMAKTADIR !

############################
# Descriptive Statistics (Betimsel İstatistikler)
############################

df = sns.load_dataset("tips")
df.describe().T

############################
# Confidence Intervals (Güven Aralıkları)
############################

df = sns.load_dataset("tips")
df.head()
df.describe().T
# Total_bill değişkeninin güven aralıklarını hesaplayarak en az ve en köt beklenti hesaplanabilir
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest


sms.DescrStatsW(df["total_bill"]).tconfint_mean()
# çıktı: (18.663331704358473, 20.908553541543167)
# buradaki yorum: %95 doğruluk payıyla müşteriler 18.66 ile 20.9 arasında ortalama bir değer bırakacaklardır !
sms.DescrStatsW(df["tip"]).tconfint_mean()
# çıktı: (2.823799306281821, 3.1727580707673595). bahşiş ortalama güven aralığı
# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()

######################################################
# Correlation (Korelasyon)
######################################################


# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?
df = sns.load_dataset("tips")
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]
df.plot.scatter("tip", "total_bill")
plt.show(block=True)
sns.scatterplot(df, x="tip", y="total_bill")
plt.show(block=True)

df["tip"].corr(df["total_bill"])  # iki değişken araındaki korelasyon hesaplanmıştır
# çıktı: 0.5766634471096374. burada tips ile toal bill arasında opozitif orta şiddettte bir korelasyon gözlenmiştir !
corr = df[num_cols].corr()  # keşifçi fonksionel veri analizinden alınan bu kodda ise nümerik değişkenlerin birbirleriyle
# arasında olan korelasyon incelenmiştir !

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2.2 numara. Varyans homojenliği sağlanmıyorsa 2.1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistiksel Olalarak Anlamlı Bir Fark var mı?
############################

df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})
# çıktı:
# smoker
# Yes    3.00871
# No     2.99185
# NOT: burada sigara içmeme durumuna göre 2 grubun ortalaması alındığında, 2 grubun ödediği total_bill değerinde
# fark olduğu görünüyor. Bu farkın şans eseri ortaya çıkıp çıkmadığını inceleyeceğiz !
####################################
# 1. HİPOTEZİ KUR
####################################
# HO: M1 = M2 (Sigara İçenler ile İçmeyenlerin ödediği ortalama total_bill miktarında farklılık yoktur)
# H1: M1 != M2
####################################
# 2. Varsayım Kontrolü
####################################
#   - 1. Normallik Varsayımı : BİR DEĞİŞKENİN DAĞILIMININ NORMAL OLUP OLMADĞININ HİPOTEZ TESTİDİR !
#   - 2. Varyans Homojenliği
####################################
# 2.1 NORMAL DAĞILIM HİPOTEZİNİ KURUYORUZ
####################################
# HO: NORMAL DAĞILIM VARSAYIMI SAĞLANMAKTADIR
# H1: .... SAĞLANMAMAKTADIR

tes_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# The Shapiro-Wilk test tests the null hypothesis that the data was drawn from a normal distribution.
# SHAPIRO TESTİNDE, FORMÜLE GİRİLEN GRUBUN DAĞILIMININ NORMAL OLUP OLADIĞINI TEST EDER.
# çıktı: Test Stat =  0.9367, p-value = 0.0002
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# p-value = 0.0002 < 0.05 olduğundan sigara içen grubunun ödediği total_bill miktarında normallik dağılımı sağlanmamakdır.

tes_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# çıktı: Test Stat =  0.9045, p-value = 0.0000
# p-value = 0.0000 < 0.05 olduğundan sigara içmeyen grubunun ödediği total_bill miktarında da normallik dağılımı sağlanmamakdır.
####################################
# 2.2 VARYANS HOMOJENLİĞİ VARSAYIMI HİPOTEZİNİ KURUYORUZ
####################################
# HO: VARYANSLAR HOMOJENDİR
# H1: VARYANSLAR HOMOJEN DEĞİLDİR.

# NOT: VARYANS HOMOJENLİĞİNİ SINAMAK İÇİN LEVENE TESTİ KULLANILIR.LEVENE TESTİNDE İKİ GRUBUNDA BİLGİLERİ GÖNDERİLEİR VE VARYANS HOMOJENLİĞİNE BAKILIR.
tes_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                          df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# ÇIKTI: Test Stat =  4.0537, p-value = 0.0452
# p-value = 0.0.0452 < 0.05 olduğundan varyansların homejn dağıldığı hipotezi REDDEDİLİR.
####################################
# 3. ve 4. Hipotezin Uygulanması ve Yorumlanması
####################################
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


# NOT: EĞER VARSAYIMLAR SAĞLANSAYDI AŞAĞIDAKİ GİBİ TEST YAPILACAKTI
####################################
# 3.1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# NOT: örnek olarak yapıyoruz. total_bill değişkeninde dağılım normal değil ve de varyanslar homojen değildir.
####################################

test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"], equal_var=True)
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

# ÖNEMLİ BİLGİ:
# 1 NORMALLİK VARSAYIMI SAĞLANIYORSA KULLANILIR
# 2 NORMALLİK VARSAYIMI VE VARYANS HOMOJENDLİĞİ SAĞLANIYORSA KULLANILIR
# 3 NORMALLİK VARSAYIMI SAĞLANIYOR FAKAT VARYANS HOMOJENDLİĞİ SAĞLANMIYOR DA KULLANILIR( equal_var=False) işaretlenir !


# çıktı: Test Stat =  4.0537, p-value = 0.1820
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# p-value = 0.1820 > 0.05 olduğundan sigara içmeyen grubunun ödediği total_bill miktarında da normallik dağılımı sağlanmamakdır.

# ÇOK ÖENMLİ. BU HİPOTEZ TESTLERİ SONUCUNDA H0 HİPOTEZİNİ YA KABUL EDERİZ. YA DA REDDEDERİZ.
# H1 HİPOTEZİNİ KABUL ETME DURUMU SÖZ KONUSU DEĞİLDİR! ÇÜKNKÜ SADECE H0 I KABUL EDİP ETMEME DURUMUNDA YAPACAĞIMIZ
# HATA MİKTARINI 0.05 DEĞERİNİ BİLİR. H1 İÇİN BİR ŞEY DİYEMİYORUZ

####################################
# 3.2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test) - Sadece normal dağılım varsayımı sağlamsa da olur!
####################################

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

# çıktı: Test Stat =  4.0537, p-value = 0.3413
# p-value = 0.3413 > 0.05 olduğundan sigara içEN İLE İÇMENEYEN GRUPLAR ARASIDNA İSTATİSTİKİ OLARAK BİR FARK YOKTUR!
# MEAN OLARAK HESAPLANAN ORTALAM DEĞERLER ARASIDAKİ FARKLILIK ŞANS ESERİ ORTAYA ÇIKMIŞTIR!
# HO: M1 = M2 (Sigara İçenler ile İçmeyenlerin ödediği ortalama total_bill miktarında farklılık yoktur) HİPOTEZİNDEKİ
# H0 REDDEDİLEMEZ

############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################
df = sns.load_dataset("titanic")
df.head()
df.groupby("sex").agg({"age": "mean"})

####################################
# 1. HİPOTEZİ KUR
####################################
# HO: M1 = M2 (titanic deki kadın ve erkek yolcuların yaş ortalamaları arasında istatistiki olarak fark yoktur)
# H1: M1 != M2

# 2.1.
# HO: M1 = M2 :İLGİLECEK GRUBUN YAŞ DAĞILIMI NORMAL DAĞILMIŞTIR
# H1: M1 != M2: .... VARDIR

tes_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# ÇIKTI: Test Stat =  0.9747, p-value = 0.0000
# PVALUE < 0.05 OLDUĞUNDAN ERKEKLERİN YAŞLARININ NORMAL DAĞILIMI H0 REDDEDİLİR. YANİ NORMAL DAĞILMAMIŞTIR
tes_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# ÇIKTI: Test Stat =  0.9848, p-value = 0.0071
# PVALUE < 0.05 OLDUĞUNDAN KADINLARIN YAŞLARININ NORMAL DAĞILIMI H0 REDDEDİLİR. YANİ NORMAL DAĞILMAMIŞTIR

# NOT: BURADA NORMLADE DİREKT OLARAK NON-PARAMETRİK TESTE GEÇİLMELİDİR. AMA YİNE DE ÖRNEK OLMASI AÇISINDAN VARYANS
# HOMOJENLİĞİNİ KONTROL EDİYORUZ.
# DAĞILIMIN NORMLA OLUP VARYANSLARIN KONTROL EDİLMESİNİN NEDENİ, PARAMETRİK TESTTE KULLANILAN
# ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],df.loc[df["smoker"] == "No", "total_bill"], equal_var=True)
# FORMÜLÜNDE eeual_var parametresinin doğru olup olmadığını belirlemektir.
#
# 2.2.
# HO: M1 = M2 : İKİ GRUBUNDA VARYANSLARI HOMOJENDİR
# H1: M1 != M2: .... HOMOJEN DEĞİLDİR
tes_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                          df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# çıktı: Test Stat =  0.0013, p-value = 0.9712
# PVALUE - 0.9712 > 0.05 OLDUĞUNDAN İKİ GRUBUN VARYANSI HOMOJENDİR.

# NORMAL DAĞILIM VARSAYIMI SAĞLANMADIĞI İÇİN NON-PARAMETRİK TESTE DÜŞÜYORUZ !
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "male", "age"].dropna(),
                                 df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

# buradaki amaç yaş ile diyabet arasındaki ilişkiyi istatistiksel olarak incelemektir

df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/diabetes.csv")
df.head()
df.shape
df.groupby("Outcome").agg({"Age": "mean"})
# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2.2 numara. Varyans homojenliği sağlanmıyorsa 2.1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

# 1. H0: M1 = M2 (diyabet olan ve olmayan hastaların yaş ort. arasında fark yoktur)
# 2. H1: M1 != M2

# 2.1 H0:M1 = M2 İLGİLİ GRUPLARIN YAŞLARI NORMAL OLARAK DAĞILMIŞRI
#       H1: M1 != M2 ... YAŞLAR NORMAL DAĞILMAMIŞTIR

tes_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# p-value = 0.0000 < 0.05 old. H0 REDDEDİLİR.

tes_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# p-value = 0.0000 < 0.05 old. H0 REDDEDİLİR.

# NOT: HER İKİ GRUBUNDA YAŞLARI NORMAK DAĞILAMIŞITRI . NON-PARAMETRİK TEST !

# 2.2 H0:M1 = M2 HER İKİ GRUBUN DA VARYANSLARI HOMOJENDİR
#       H1: M1 != M2 ... HOMOJEN DEĞİLDİR

tes_stat, pvalue = levene(df.loc[df["Outcome"] == 0, "Age"],
                          df.loc[df["Outcome"] == 1, "Age"])

# pvalue - 0.13618 > 0.05 olduğundan H0 reddedilemez

# normal dağılımı varsayımı sağlanmadığından non-prametrik test, mannwithneyu testi yapıyoruz
# non-parametrik: medyan kıyaslama

tes_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 0, "Age"],
                                df.loc[df["Outcome"] == 1, "Age"])
# pvalue - 1.1422001179619007e-17 < 0.05 olduğunda H0 Reddedilir.
# H0: M1 = M2 (diyabet olan ve olmayan hastaların yaş ort. arasında fark yoktur)
# YANİ ŞANSA YER BIRAKMAYACAK ŞEKİLDE DİYABET OLANLARIN YAŞLARI(YAŞ ORTALAMSI YÜKSEK OLDUĞUNDAN) DAHA YÜKSEKTİR.
###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("measurement_problems/datasets_for_measurement_problems/course_reviews.csv")
df.head()

# BURADA KURSUN BELİRLİ BİR ORANININ İZLEYENLERİN VERDİĞİ PUANLAR ARASINDA İSTATATİSTİKSEL OLARAK ANLAMLI BİR FARKLILIK VAR MI NA BAKIYORUZ!
# H0: M1 = M2 (... iki grup rating ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df.loc[df["Progress"] > 75, "Rating"].mean()
df.loc[df["Progress"] < 10, "Rating"].mean()
# H0:M1 = M2 İLGİLİ GRUPLARIN VERDİĞİ RATING ORANLARI NORMAL DAĞILMIŞTIR
#       H1: M1 != M2 ... İLGİLİ GRUPLARIN VERDİĞİ RATING ORANLARI NORMAL DAĞILMAMIŞTIR


tes_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

tes_stat, pvalue = shapiro(df.loc[df["Progress"] < 10, "Rating"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

# SONUÇ: HER İKİ GRUP İÇİNDE H0(M1 = M2 İLGİLİ GRUPLARIN VERDİĞİ RATING ORANLARI NORMAL DAĞILMIŞTIR) REDDEDİLİR. GO TO NON-PARAMETRİK TEST

tes_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] > 75, "Rating"],
                                df.loc[df["Progress"] < 25, "Rating"])
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

# SONUÇ: ANA HİPOTEZ H0 REDDEDİLİR.
# H0: M1 = M2 (... iki grup rating ortalamaları arasında ist ol.anl.fark yoktur.)
# NOT: BURAYA KADARKİ OLAN SÜREÇLERDE DAHA ÖNCE GRUP ORTALAMALARI VE MEDYANLARI KIYASLANDI

######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################
# BU TESTLER İKİ ORAN KIYASI İÇİN KULLANILIR!
# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

# NOT: KULLANILACAK formüllerde np.array beklenmektedir.
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
# çıktı: Out[100]: (3.7857863233209255, 0.0001532232957772221)
# pvalue -  0.0001532232957772221 < 0.05 olduğundan H0 Reddedilir

basari_sayisi / gozlem_sayilari

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")

df.loc[df["sex"] == "female", "survived"].mean()
df.loc[df["sex"] == "male", "survived"].mean()

basari_sayisi = np.array([df.loc[(df["sex"] == "female"), "survived"].sum(),
                          df.loc[(df["sex"] == "male"), "survived"].sum()])

gozlem_sayilari = np.array([df.loc[(df["sex"] == "female"), "survived"].count(),
                            df.loc[(df["sex"] == "male"), "survived"].count()])

tes_stat, pvalue = proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))
# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur
# p-value = 0.0000 < 0.05 olduğundan H0 Reddedilir. yaşlar arasında fark vardır

######################################################
# ANOVA (Analysis of Variance)
######################################################
# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.

# NOT: BURADA SADECE 3 GURBUNU ORTALAMSINI KARŞILAŞTIRARAK H0 İNCELİYOR OLACAĞIZ

# 1. HİPOTEZLERİ KUR
# H0: M1=M2=M3=M4 ( TIPS VERİ SETİNDE GÜNLERİN( GRUPLARIN) ORTALAMALARI(ORTALAMA KAZANÇLARI) ARASINDA FARK YOKTUR)
# H1: EN AZ BİRİ FARKLIDIR
df = sns.load_dataset("tips")
df.head()

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2.2 numara. Varyans homojenliği sağlanmıyorsa 2.1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.
# 2.VARSAYIMLARIN KONTROLÜ
# 2.1. nORMALLİK VARSAYIMI
# 2.2. VARYANS HOMOJENLİĞİ VARSAYIMI

# VARSAYIM SAĞLANIYORSA ONE WAY ANOVA TESTİ ( PARAMETRİK 2 DEN FAZLA GRUP KARŞILAŞTIRMA TESTİ)
# VARSAYIM SAĞLANMIYORSA KRUSKAL TESTİ (NON-PARAMETRİK 2 DEN FAZLA GRUP KARŞILAŞTIRMA TESTİ)

# 2.1.1 H0: NORMALLİK DAĞILIMI SAĞLANMAKTDAIR

df.groupby("day").agg({"total_bill": "mean"})

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]  # 0.indexte stats değeri var
    print(group, "p-value: %.4f" % pvalue)
# çıktı:
# Sun p-value: 0.0036
# Sat p-value: 0.0000
# Thur p-value: 0.0000
# Fri p-value: 0.0409

# GÜNLÜK TÜM PVALUE DEĞERLERİNDE p-value < 0.05 olduğundan "H0: NORMALLİK DAĞILIMI SAĞLANMAKTDAIR" REDDEDİLİR.
# 2.1.2 H0: VARYANS HOMOJEN DAĞILMIŞTIR

tes_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                          df.loc[df["day"] == "Sat", "total_bill"],
                          df.loc[df["day"] == "Thur", "total_bill"],
                          df.loc[df["day"] == "Fri", "total_bill"])

print("Test Stat = % .4f, p-value = %.4f" % (tes_stat, pvalue))

# sonuç: p-value = 0.5741 > 0.05 dolayısıyla "H0: VARYANS HOMOJEN DAĞILMIŞTIR" REDDEDİLEMEZ.


# 3. ve 4. HİPOTEZ TESTİ

# EĞER "H0: NORMALLİK DAĞILIMI SAĞLANMAKTDAIR" SAĞLANSAYDI ŞUNLARI YAPACAKTIK:

# PARAMETRİK ANOVA TESTİ - ONE WAY ANOVA
f_oneway(df.loc[df["day"] == "Sun", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"])
# sonuç: pvalue=0.04245383328951916 > 0.05 oldğundan H0 reddedilir.

# NONPARAMETRİK ANOVA TESTİ - KRUSKAL TESTİ
kruskal(df.loc[df["day"] == "Sun", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"])
# sonuç: pvalue=0.015433008201042065 < 0.05 olduğundan H0 Reddedilir. Fark vardır.
# H0: M1=M2=M3=M4 ( TIPS VERİ SETİNDE GÜNLERİN( GRUPLARIN) ORTALAMALARI(ORTALAMA KAZANÇLARI) ARASINDA FARK YOKTUR)

# GRUPLAR ARASINDAKİ FARK HANGİSİNDEN KAYNAKLANMAKTADIR ?

from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())
# çıktı:
# Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ====================================================
# group1 group2 meandiff p-adj   lower   upper  reject
# ----------------------------------------------------
#    Fri    Sat   3.2898 0.4541 -2.4799  9.0595  False
#    Fri    Sun   4.2584 0.2371 -1.5856 10.1025  False
#    Fri   Thur   0.5312 0.9957 -5.4434  6.5057  False
#    Sat    Sun   0.9686 0.8968 -2.6088   4.546  False
#    Sat   Thur  -2.7586 0.2374 -6.5455  1.0282  False
#    Sun   Thur  -3.7273 0.0668 -7.6264  0.1719  False
# ----------------------------------------------------

## AB TESTLERİNDE Fayda-maliyet dengesinde yapmak istediğim çalışma kayda değer mi bunun yanıtını almaya çalışıyoruz.
# AB TESTLERİNDE ESAS AMAÇ İKİ GRUBUN ORTALAMSI ARASINDAKİ FARKIN, ŞANS ESERİ ÇIKIP ÇIKMADIĞINI BULMAKTIR !!!!

# ELİMİZDE KABUL EDİLEBİLİR 1 HATA MİKTARI VAR VE BU HATA MİKATIRINI