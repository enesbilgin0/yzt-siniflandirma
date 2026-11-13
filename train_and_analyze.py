# -*- coding: utf-8 -*-
"""
train_and_analyze.py
--------------------
Kalp Hastalığı Tahmini (Evet mi, Hayır mı?) projesinin ANA betiği.

Bu betik sırasıyla şunları yapar:
  1. Heart Disease UCI (Cleveland) veri setini internetten indirir -> data/heart.csv
     (indirilemezse yerleşik gerçekçi bir örnek veriye düşer)
  2. Veriyi tanıtır ve temizler (eksik değer, sınıf dengesi vb.)
  3. 5 adet analiz grafiği üretir -> plots/
  4. 3 modeli (Logistic Regression, Decision Tree, Random Forest) eğitip karşılaştırır
  5. Confusion Matrix ve model karşılaştırma grafiklerini kaydeder
  6. Modelin yanlış tahmin ettiği örnekleri analiz eder
  7. En iyi modeli ve yardımcı bilgileri model.pkl olarak kaydeder

Çalıştırmak için:  python train_and_analyze.py
"""

import os
import ssl
import warnings
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Ekran olmadan (sunucuda) grafik üretebilmek için
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import certifi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  Genel ayarlar: klasörler, sütun isimleri ve modern renk paleti
# --------------------------------------------------------------------------- #
DATA_DIR = "data"
PLOTS_DIR = "plots"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Modern, göze hoş gelen bir seaborn teması
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.size"] = 11

# 14 sütunun sıralı ismi (UCI ham dosyasında başlık yoktur)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]

# Sütunların Türkçe açıklamaları (arayüz ve raporlarda kullanılır)
COLUMN_TR = {
    "age": "Yaş",
    "sex": "Cinsiyet (1=Erkek, 0=Kadın)",
    "cp": "Göğüs Ağrısı Tipi (0-3)",
    "trestbps": "Dinlenme Kan Basıncı (mm Hg)",
    "chol": "Kolesterol (mg/dl)",
    "fbs": "Açlık Kan Şekeri > 120 (1=Evet, 0=Hayır)",
    "restecg": "Dinlenme EKG Sonucu (0-2)",
    "thalach": "Maksimum Kalp Atış Hızı",
    "exang": "Egzersize Bağlı Anjina (1=Var, 0=Yok)",
    "oldpeak": "ST Depresyonu (oldpeak)",
    "slope": "ST Eğimi (0-2)",
    "ca": "Boyanan Ana Damar Sayısı (0-3)",
    "thal": "Talasemi (1=Normal, 2=Sabit, 3=Geri Dönüşlü)",
    "target": "Kalp Hastalığı (1=EVET/Hasta, 0=HAYIR/Sağlıklı)",
}


# --------------------------------------------------------------------------- #
#  1. VERİ SETİNİ İNDİRME / YÜKLEME
# --------------------------------------------------------------------------- #
def _indir(url: str) -> str:
    """Verilen URL'den metni indirir.
    macOS'ta sık görülen sertifika hatasını, güvenliği bozmadan, 'certifi'
    güvenilir kök sertifikalarını kullanan doğrulamalı bir SSL bağlamıyla aşar."""
    try:
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    return urlopen(url, timeout=20, context=ctx).read().decode("utf-8", errors="ignore")


def _sentetik_veri(n: int = 303) -> pd.DataFrame:
    """İnternet yoksa: istatistiksel olarak gerçekçi, yedek bir veri seti üretir.
    (Gerçek veri her zaman tercih edilir; bu sadece güvenli bir yedektir.)"""
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.normal(54, 9, n).clip(29, 77).round()
    sex = rng.binomial(1, 0.68, n)
    cp = rng.integers(0, 4, n)
    trestbps = rng.normal(131, 17, n).clip(94, 200).round()
    chol = rng.normal(246, 52, n).clip(126, 564).round()
    fbs = rng.binomial(1, 0.15, n)
    restecg = rng.integers(0, 2, n)
    thalach = rng.normal(149, 23, n).clip(71, 202).round()
    exang = rng.binomial(1, 0.33, n)
    oldpeak = rng.gamma(1.3, 0.8, n).clip(0, 6.2).round(1)
    slope = rng.integers(0, 3, n)
    ca = rng.integers(0, 4, n)
    thal = rng.integers(1, 4, n)

    # Hastalık olasılığını klinik mantıkla kur (yaş, anjina, cp, oldpeak, damar vb.)
    risk = (
        0.03 * (age - 54)
        + 0.9 * exang
        - 0.55 * cp
        + 0.45 * oldpeak
        + 0.6 * ca
        - 0.02 * (thalach - 149)
        + 0.5 * sex
        + 0.4 * (thal == 3)
    )
    prob = 1 / (1 + np.exp(-risk))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal, "target": target,
    })
    return df


def veri_yukle() -> pd.DataFrame:
    """Veri setini önce yerelden, sonra internetten, en son yedek veriden yükler."""
    yerel = os.path.join(DATA_DIR, "heart.csv")
    if os.path.exists(yerel):
        print(f"[BILGI] Veri yerelden okunuyor: {yerel}")
        return pd.read_csv(yerel)

    # Denenecek güvenilir açık kaynak adresleri (sırayla)
    kaynaklar = [
        # 1) Zaten başlıklı, 303x14 'heart.csv' (target 1=hasta)
        ("csv", "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Heart%20Disease.csv"),
        # 2) UCI orijinal Cleveland ham dosyası (başlıksız, '?' içerir, num 0-4)
        ("uci", "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"),
    ]

    for tur, url in kaynaklar:
        try:
            print(f"[BILGI] Veri indiriliyor: {url}")
            ham = _indir(url)
            from io import StringIO

            if tur == "uci":
                df = pd.read_csv(StringIO(ham), header=None, names=COLUMNS)
                # UCI'de hedef 0-4 arası 'num'; >0 ise hastalık VAR demektir -> 1
                df["target"] = (pd.to_numeric(df["target"], errors="coerce") > 0).astype(int)
            else:
                df = pd.read_csv(StringIO(ham))
                df.columns = [c.strip().lower() for c in df.columns]
                # Bazı sürümlerde hedef sütunu 'num' olabilir
                if "target" not in df.columns and "num" in df.columns:
                    df = df.rename(columns={"num": "target"})
                df = df[COLUMNS]  # sütun sırasını sabitle

            if len(df) > 50:  # mantıklı bir veri geldiyse kabul et
                df.to_csv(yerel, index=False)
                print(f"[BASARI] Veri kaydedildi: {yerel}  ({len(df)} satır)")
                return df
        except Exception as e:
            print(f"[UYARI] İndirme başarısız ({url}): {e}")

    print("[UYARI] İnternetten indirilemedi -> yerleşik yedek veri üretiliyor.")
    df = _sentetik_veri()
    df.to_csv(yerel, index=False)
    print(f"[BASARI] Yedek veri kaydedildi: {yerel}  ({len(df)} satır)")
    return df


# --------------------------------------------------------------------------- #
#  2. VERİYİ TANIMA VE TEMİZLEME
# --------------------------------------------------------------------------- #
def veri_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """Eksik/hatalı değerleri temizler, tipleri düzeltir, sınıf dengesini raporlar."""
    print("\n" + "=" * 60)
    print("VERİYİ TANIMA VE TEMİZLEME")
    print("=" * 60)

    # '?' gibi değerleri NaN'a çevir ve sayısal tipe geç
    df = df.replace("?", np.nan)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    eksik = df.isna().sum().sum()
    print(f"Toplam eksik değer sayısı: {eksik}")
    if eksik:
        # Eksikleri sütun medyanı ile doldur (sağlık verisinde güvenli bir seçim)
        df = df.fillna(df.median(numeric_only=True))
        print("-> Eksik değerler sütun medyanı ile dolduruldu.")

    # Tekrar eden satırları at
    once = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) != once:
        print(f"-> {once - len(df)} adet tekrar eden satır kaldırıldı.")

    df["target"] = df["target"].astype(int)

    # Sınıf dengesi raporu
    n_evet = int((df["target"] == 1).sum())
    n_hayir = int((df["target"] == 0).sum())
    print(f"\nToplam hasta kaydı : {len(df)}")
    print(f"  EVET  (1 - Kalp Hastası) : {n_evet}  (%{100*n_evet/len(df):.1f})")
    print(f"  HAYIR (0 - Sağlıklı)     : {n_hayir}  (%{100*n_hayir/len(df):.1f})")
    denge = "DENGELİ" if 0.4 <= n_evet / len(df) <= 0.6 else "HAFİF DENGESİZ"
    print(f"  Sınıf dengesi: {denge}")

    return df


# --------------------------------------------------------------------------- #
#  3. GÖRSELLEŞTİRME (5 GRAFİK)
# --------------------------------------------------------------------------- #
def grafikleri_uret(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("GÖRSELLEŞTİRME - 5 grafik plots/ klasörüne kaydediliyor")
    print("=" * 60)

    etiket = {0: "HAYIR (Sağlıklı)", 1: "EVET (Hasta)"}
    renkler = ["#2ecc71", "#e74c3c"]  # yeşil=sağlıklı, kırmızı=hasta

    # --- Grafik 1: Sınıf Dağılımı ---
    fig, ax = plt.subplots(figsize=(7, 5))
    sayim = df["target"].map(etiket).value_counts()
    sns.barplot(x=sayim.index, y=sayim.values, palette=renkler, ax=ax)
    for i, v in enumerate(sayim.values):
        ax.text(i, v + 1, str(v), ha="center", fontweight="bold")
    ax.set_title("Grafik 1 — Sınıf Dağılımı (Kaç Evet / Kaç Hayır?)", fontweight="bold")
    ax.set_ylabel("Hasta Sayısı")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "1_sinif_dagilimi.png"))
    plt.close(fig)

    # --- Grafik 2: Cinsiyete Göre Kalp Hastalığı Oranı ---
    fig, ax = plt.subplots(figsize=(7, 5))
    tmp = df.copy()
    tmp["Cinsiyet"] = tmp["sex"].map({0: "Kadın", 1: "Erkek"})
    tmp["Durum"] = tmp["target"].map(etiket)
    sns.countplot(data=tmp, x="Cinsiyet", hue="Durum", palette=renkler, ax=ax)
    ax.set_title("Grafik 2 — Cinsiyete Göre Kalp Hastalığı", fontweight="bold")
    ax.set_ylabel("Hasta Sayısı")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "2_cinsiyet_hastalik.png"))
    plt.close(fig)

    # --- Grafik 3: Göğüs Ağrısı Tipine Göre Hastalık ---
    fig, ax = plt.subplots(figsize=(7, 5))
    tmp = df.copy()
    tmp["Durum"] = tmp["target"].map(etiket)
    sns.countplot(data=tmp, x="cp", hue="Durum", palette=renkler, ax=ax)
    ax.set_title("Grafik 3 — Göğüs Ağrısı Tipine Göre Hastalık", fontweight="bold")
    ax.set_xlabel("Göğüs Ağrısı Tipi (0=Tipik, 1=Atipik, 2=Non-anjinal, 3=Asemptomatik)")
    ax.set_ylabel("Hasta Sayısı")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "3_gogus_agrisi_etkisi.png"))
    plt.close(fig)

    # --- Grafik 4: Yaş & Maksimum Nabzın Etkisi (Scatter) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    tmp = df.copy()
    tmp["Durum"] = tmp["target"].map(etiket)
    sns.scatterplot(data=tmp, x="age", y="thalach", hue="Durum",
                    palette=renkler, s=70, alpha=0.8, ax=ax)
    ax.set_title("Grafik 4 — Yaş ve Maksimum Kalp Atış Hızı", fontweight="bold")
    ax.set_xlabel("Yaş")
    ax.set_ylabel("Maksimum Kalp Atış Hızı (thalach)")
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "4_yas_ve_maks_nabiz.png"))
    plt.close(fig)

    # --- Grafik 5: Korelasyon Isı Haritası ---
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Grafik 5 — Özelliklerin Korelasyon Isı Haritası", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "5_korelasyon_haritasi.png"))
    plt.close(fig)

    print("-> 5 analiz grafiği kaydedildi.")


# --------------------------------------------------------------------------- #
#  4. MODELLEME VE KARŞILAŞTIRMA
# --------------------------------------------------------------------------- #
def modelleri_egit(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("MODELLEME VE KARŞILAŞTIRMA (3 Model)")
    print("=" * 60)

    X = df.drop(columns=["target"])
    y = df["target"]

    # %80 eğitim / %20 test, sınıf oranını koruyarak (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Özellikleri ölçekle (özellikle Lojistik Regresyon için önemli)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    modeller = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    sonuclar = []
    egitilmis = {}
    for ad, model in modeller.items():
        model.fit(X_train_s, y_train)
        tahmin = model.predict(X_test_s)
        sonuclar.append({
            "Model": ad,
            "Accuracy": accuracy_score(y_test, tahmin),
            "Precision": precision_score(y_test, tahmin, zero_division=0),
            "Recall": recall_score(y_test, tahmin, zero_division=0),
            "F1-Score": f1_score(y_test, tahmin, zero_division=0),
        })
        egitilmis[ad] = model
        print(f"\n{ad}")
        print(f"  Accuracy : {sonuclar[-1]['Accuracy']:.3f}")
        print(f"  Precision: {sonuclar[-1]['Precision']:.3f}")
        print(f"  Recall   : {sonuclar[-1]['Recall']:.3f}  <-- SAĞLIKTA EN KRİTİK METRİK")
        print(f"  F1-Score : {sonuclar[-1]['F1-Score']:.3f}")

    sonuc_df = pd.DataFrame(sonuclar).round(3)

    # En iyi modeli RECALL'e göre seç (yanlış 'Hayır'ı en aza indirmek istiyoruz)
    en_iyi_ad = sonuc_df.sort_values("Recall", ascending=False).iloc[0]["Model"]
    en_iyi_model = egitilmis[en_iyi_ad]
    print(f"\n[SECIM] En iyi model (Recall'e göre): {en_iyi_ad}")

    # --- Grafik 6: Model Karşılaştırması ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cizim = sonuc_df.melt(id_vars="Model", var_name="Metrik", value_name="Deger")
    sns.barplot(data=cizim, x="Metrik", y="Deger", hue="Model", palette="Set2", ax=ax)
    ax.set_title("Grafik 6 — Model Karşılaştırması", fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_xlabel("")
    ax.legend(title="", loc="lower right")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.text(p.get_x() + p.get_width() / 2, h + 0.01, f"{h:.2f}",
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "6_model_karsilastirma.png"))
    plt.close(fig)

    # --- Confusion Matrix (en iyi model için) ---
    tahmin = en_iyi_model.predict(X_test_s)
    cm = confusion_matrix(y_test, tahmin)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["HAYIR (0)", "EVET (1)"],
                yticklabels=["HAYIR (0)", "EVET (1)"], ax=ax)
    ax.set_title(f"Confusion Matrix — {en_iyi_ad}", fontweight="bold")
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    # FN hücresini vurgula (gerçek=EVET, tahmin=HAYIR -> en tehlikeli hata)
    ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor="red", lw=3))
    ax.text(0.5, 1.75, "TEHLİKELİ: Yanlış HAYIR (FN)", color="red",
            ha="center", fontsize=8, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close(fig)

    print("-> Model karşılaştırma ve confusion matrix grafikleri kaydedildi.")

    paket = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "features": list(X.columns),
    }
    return en_iyi_model, en_iyi_ad, sonuc_df, paket


# --------------------------------------------------------------------------- #
#  5. YANLIŞ TAHMİN ANALİZİ
# --------------------------------------------------------------------------- #
def hata_analizi(model, paket) -> pd.DataFrame:
    """Modelin yanıldığı somut örnekleri bulur ve kurallı Türkçe yorum üretir."""
    print("\n" + "=" * 60)
    print("YANLIŞ TAHMİN ANALİZİ (Model neden yanıldı?)")
    print("=" * 60)

    X_test, y_test = paket["X_test"], paket["y_test"]
    X_test_s = paket["scaler"].transform(X_test)
    tahmin = model.predict(X_test_s)

    hatali = X_test[tahmin != y_test.values].copy()
    hatali["Gercek"] = y_test.values[tahmin != y_test.values]
    hatali["Tahmin"] = tahmin[tahmin != y_test.values]

    def yorumla(satir):
        g, t = satir["Gercek"], satir["Tahmin"]
        if g == 1 and t == 0:  # False Negative - en tehlikeli
            return ("TEHLİKELİ (Yanlış HAYIR / FN): "
                    f"Hasta {int(satir['age'])} yaşında, maks. nabzı {int(satir['thalach'])} "
                    f"ve göğüs ağrısı tipi {int(satir['cp'])}. Değerler 'normal'e yakın göründüğü "
                    "için model 'Sağlıklı' dedi, oysa hasta gerçekte kalp rahatsızı.")
        else:  # False Positive
            return ("Yanlış EVET (FP): "
                    f"Hasta {int(satir['age'])} yaşında, kolesterol {int(satir['chol'])} "
                    f"ve oldpeak {satir['oldpeak']}. Riskli değerler yüzünden model 'Hasta' dedi, "
                    "oysa kişi sağlıklı. (Sonuç: sadece gereksiz ek kontrol.)")

    if len(hatali):
        hatali["Yorum"] = hatali.apply(yorumla, axis=1)
    else:
        hatali["Yorum"] = []

    # Önce en tehlikeli hataları (FN) göster
    hatali = hatali.sort_values("Gercek", ascending=False)
    goster = hatali.head(4)
    print(f"Test setinde toplam {len(hatali)} yanlış tahmin var. İlk örnekler:\n")
    for _, s in goster.iterrows():
        print("  - " + s["Yorum"])

    return hatali


# --------------------------------------------------------------------------- #
#  ANA AKIŞ
# --------------------------------------------------------------------------- #
def main():
    print("KALP HASTALIĞI TAHMİNİ — EĞİTİM VE ANALİZ BETİĞİ\n")

    df = veri_yukle()
    df = veri_temizle(df)
    grafikleri_uret(df)
    en_iyi_model, en_iyi_ad, sonuc_df, paket = modelleri_egit(df)
    hatali = hata_analizi(en_iyi_model, paket)

    # --- Mülakat / Jüri sorusu vurgusu ---
    print("\n" + "#" * 60)
    print("# KRİTİK SORU: Yanlış 'HAYIR' mı daha kötü, yanlış 'EVET' mi?")
    print("#" + "-" * 58)
    print("# Yanlış 'HAYIR' (False Negative): Hastaya 'sağlıklısın' demek ->")
    print("#   tedavi gecikir, HAYATİ RİSK doğar. EN TEHLİKELİ HATA budur.")
    print("# Yanlış 'EVET' (False Positive): Sağlıklıya 'hastasın' demek ->")
    print("#   sadece ek doktor kontrolü / test maliyeti.")
    print("# SONUÇ: Bu yüzden RECALL (Duyarlılık) bizim için en kritik metriktir.")
    print("#" * 60)

    # --- Modeli ve ek bilgileri kaydet ---
    kayit = {
        "model": en_iyi_model,
        "model_adi": en_iyi_ad,
        "scaler": paket["scaler"],
        "features": paket["features"],
        "column_tr": COLUMN_TR,
        "sonuc_df": sonuc_df,
        "hatali_ornekler": hatali,
        "sinif_dagilimi": df["target"].value_counts().to_dict(),
        "veri_ozeti": df.describe(),
    }
    joblib.dump(kayit, MODEL_PATH)
    print(f"\n[BASARI] En iyi model ve analizler kaydedildi -> {MODEL_PATH}")
    print("\nTüm adımlar tamamlandı. Arayüz için:  streamlit run app.py")


if __name__ == "__main__":
    main()
