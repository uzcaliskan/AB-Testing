import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 800)
pd.set_option("display.expand_frame_repr", False)  # pd.set_option("display.width", 800) ile aynı işi yapıyor
pd.set_option("display.float_format", lambda x: "%.5f" % x)

###################################################
# Up-Down Diff Score = (up ratings) − (down ratings)
###################################################

# Review 1: 600 up 400 down total 1000
# Review 2: 5500 up 4500 down total 10000

def score_up_down_diff(up, down):
    return up - down

# Review 1 Score:
score_up_down_diff(600, 400)

# Review 2 Score
score_up_down_diff(5500, 4500)

###################################################
# Score = Average rating = (up ratings) / (all ratings)
###################################################

def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)
score_average_rating(600, 400)
score_average_rating(5500, 4500)
# Review 1: 2 up 0 down total 2
# Review 2: 100 up 1 down total 101

score_average_rating(2, 0) #çıktı : 1.0
score_average_rating(100, 1) #çıktı: 0.9900990099009901
# NOT: (100, 1) SETİ ÇOK DAHA FAZLA SAYIDA UP ALMASINA RAĞMEN (2, 0) SETİNİN GERİSİNE DÜŞMÜŞTÜR.
# BU TERSTE BİR İŞLİK VAR !!!

###################################################
# Wilson Lower Bound Score
###################################################
# BERNOULLI PARAMETRESI P İÇİN BİR GÜVEN ARALIĞI HESAPLAR VE BU GÜVEN ARALIĞININ ALT SINIRINI WLB(WILSON LOWER BOND) SCORE OLARAK KABUL EDER!

#BERNOULLI PRENSİBİ İKİLİ OLAYLARIN(YAZI-TURA VS) OLASILIĞINI HESAPLAR
def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

#MÜŞTERİLERLE İLGİLİ TÜM ETKİLEŞİMLER OLMADIĞI İÇİN LIKE-DISLIKE ORNEKLEMEYİ SEÇİYORUZ. BU ÖRNEKLEME BİLİMSEL OLARAK BİR
# GÜVEN DAYANAĞI YÜKLMEYE ÇALIŞIYORUZ
wilson_lower_bound(2, 0) # çıktı: 0.3423802275066531
wilson_lower_bound(100, 1) # çıktı: 0.9460328420055449

#ÇOK ÖENMLİ NOT: UP DOWN İLE BINARY(IKILI) RATING YAPILIYORUZ. ELİMİZDEKİ ÖRNEKLEME İLİŞKİN(BINARY ORNEKLEM)
# WLB İLE, UP RATE ORANININ İSTATİKSEL OLARAK %95 GÜVEN ORANI (oconfidence=0.95) VE %5 HATA PAYI İLE HANGİ ARALIKTA OLABİLECEĞİNİ
# BİLİYORUZ VE O ARALIĞIN ALT SINIRINI BULUYORUZ VE EN KÖTÜ SENARYODA BU REFERANS NOKTASINA TUTUNMUŞ OLUYOR.

## ÇOK ÇOK ÖNEMLİ NOT: BU WLB PROBLEMİ ÇOK ÇOK ÖENMLİDİR.

###################################################
# Case Study
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]

df = pd.DataFrame({"up": up, "down": down})
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["up"],
                                                                 x["down"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["up"],
                                                                     x["down"]), axis=1)
df["wilson_lower_bond"] = df.apply(lambda x: wilson_lower_bound(x["up"],
                                                                     x["down"]), axis=1)
df.sort_values("wilson_lower_bond", ascending=False)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)
wilson_lower_bound(280, 47)
wilson_lower_bound(376, 12)
# MIUUL İÇİN KONUŞACAK OLURSAM; KENDİ KURUMUMDA GÖZLEDİĞİM EN ÖNEMLİ ÖZELLİK İLETİŞİM VE TAKIM ÇALIŞMASINA OLAN UYGUNLUĞUN İNSANLARI ÖNE ÇIKARDIĞIDIR. PROBLEM ÇÖZME YETENEĞİ HER NE KADAR ÇOK ÖENMLİ OLSA DA BRAINSTORMING İLE DE AŞILABİLİR. ELBETTE KİŞİYİ ÖNE ÇIKARAN BİR ÖZELLİKTİR. FAKAT BENCE AĞIRLIKLANDIRMA SIRAYLA; İletişim Becerileri(%22), Takım Çalışması Becerisi(%22), Teknik yetkinlik(%21), Problem Çözme(18), Storytelling(17)
# İLETİŞİM BECELERİ İÇİN : IK ve  HR sorularına verdiği cevaplar * Mentorunun verdiği teknik, iletişim ve takım çalışması puanı
# Takım Çalışması Becerisi İÇİN:Takım çalışması etkinliklerine katılım durumu * Mentorunun verdiği teknik, iletişim ve takım çalışması puanı
# TEKNİK YETKİNLİK: Bitirme projeleri * Video izlemele/modül tamamlama durumu * Mentorunun verdiği teknik, iletişim ve takım çalışması puanı * Modül bitirme testi sonuçları
#
# PROBLEM ÇÖZME:Video izlemele/modül tamamlama durumu * Bitirme projeleri
#
# STORREY TELLING : Mentorunun verdiği teknik, iletişim ve takım çalışması puanı

wilson_lower_bound(100,2)
