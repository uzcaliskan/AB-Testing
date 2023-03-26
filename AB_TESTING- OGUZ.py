#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç



#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.

import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.5f" % x)
pd.set_option("display.expand_frame_repr", False)
#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.
df_control = pd.read_excel("measurement_problems/datasets_for_measurement_problems/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("measurement_problems/datasets_for_measurement_problems/ab_testing.xlsx", sheet_name="Test Group")
df_control.head()
df_test.head()

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

df_control.info()
df_test.info()
df_test.describe().T # Average Bidding
df_control.describe().T #Maximum Bidding

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

df = pd.concat()

df_control.columns = ['Impression_c', 'Click_c', 'Purchase_c', 'Earning_C']
df_test.columns = ['Impression_t', 'Click_t', 'Purchase_t', 'Earning_T']
df_test.head()
df_control.head()

df = pd.concat([df_test, df_control], axis=1)
df.head()
#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

## Adım 1: Hipotezi tanımlayınız.

 # H0: M1 = M1 #AVERAGE BIDDING VE MAXIMUM BIDDING PURCHASE ORT ARASINDA FARK YOKTUR
 #H1: M1 != M2 ...... FARK VARDIR

## Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df.loc[:, ["Purchase_t", "Purchase_c"]].mean()
# not: ortalamaya göre fark var görünüyor !

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)

 ## 2.1. H0: M1 = M2 - Purchase değişkeninin ortalaması normal dağılım varsayımını sağlamaktadaır
 ##      H1: M1 != M2 - ...... normal dağılmamıştır
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
test_stat, pvalue = shapiro(df.loc[:, "Purchase_t"])
print("Test Stat = % .4f, p-value = %.4f" % (test_stat, pvalue))
# p-value = 0.1541 > 0.05 olduğundan H0 Reddedilmez !
test_stat, pvalue = shapiro(df.loc[:, "Purchase_c"])
print("Test Stat = % .4f, p-value = %.4f" % (test_stat, pvalue))
## p-value =  0.5891 > 0.05 olduğundan H0 Reddedilmez

## NOT: HEM TEST HEM DE KONTROL GRUBUNUN PURCHASE ORTALAMSI NORMAL DAĞILMIŞTIR
## BUNDAN DOLAYI PARAMETRİK TEST KULLANILACAKTIR. ÖNCESİNDE VARYASNLARIN HOMOJENLİĞİ KONTROL EDİLMELİDİR !

## 2.2. H0: M1 = M2 - TEST VE KONTROL GRUPLARININ Purchase değişkeninin VARYANSLARI HOMOJENDİR / EŞİTTİR
##      H1: M1 != M2 - ...... HOMOJEN DEĞİLDİR
test_stat, pvalue = levene(df.loc[:, "Purchase_c"],
                           df.loc[:, "Purchase_t"])

print("Test Stat = % .4f, p-value = %.4f" % (test_stat, pvalue))

## pvalue=0.10828 > 0.05 olduğundan H0 REDDEDİLMEZ!
## H0: M1 = M2 - TEST VE KONTROL GRUPLARININ Purchase değişkeninin VARYANSLARI HOMOJENDİR

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz

test_stat, pvalue = ttest_ind(df.loc[:, "Purchase_c"],
                           df.loc[:, "Purchase_t"], equal_var=True)
print("Test Stat = % .4f, p-value = %.4f" % (test_stat, pvalue))

# p-value = 0.3493 > 0.05 olduğundan H0 REDDİLMEZ
# H0: M1 = M1
# #AVERAGE BIDDING VE MAXIMUM BIDDING PURCHASE ORT ARASINDA FARK YOKTUR



# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

# p-value = 0.3493 > 0.05 olduğundan H0 REDDİLMEZ
# H0: M1 = M1
# #AVERAGE BIDDING VE MAXIMUM BIDDING PURCHASE ORT ARASINDA FARK YOKTUR

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

## HER İKİ DEĞİŞKEN İÇİN DE NORMAL DAĞILIM VE VARYANSLARIN HOMOJENLİĞİ SAĞLANDIĞINDAN PARAMETRİK TEST KULLANILDI.


# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# averagebidding'inmaximumbidding'dendaha fazla dönüşüm getirip getirmediğini.
# YENİ YÖNTEM İLE ESKİ YÖNTEM ARASI FARK YOKTUR. DOLAYISIYLA EKONOMİK OLAN YÖNTEM TERCİH EDİLMELİDİR