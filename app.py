# -*- coding: utf-8 -*-
"""
app.py — Kalp Hastalığı Klinik Karar Destek Arayüzü (Streamlit)
----------------------------------------------------------------
Sunumda kullanılacak, 4 sekmeli modern arayüz:
  Sekme 1: Canlı Klinik Teşhis Simülatörü  (hasta değerleri gir -> EVET/HAYIR)
  Sekme 2: Veri Analizi ve 5 Grafik
  Sekme 3: Model Karşılaştırması & Confusion Matrix
  Sekme 4: Model Neden Yanıldı? (Hata Analizi)

Çalıştırmak için:  streamlit run app.py
(Önce 'python train_and_analyze.py' çalıştırılmış olmalıdır.)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

MODEL_PATH = "model.pkl"
PLOTS_DIR = "plots"
DATA_PATH = os.path.join("data", "heart.csv")

st.set_page_config(
    page_title="Kalp Hastalığı Klinik Karar Destek",
    page_icon="❤️",
    layout="wide",
)

# --------------------------------------------------------------------------- #
#  Sade ve şık görünüm için biraz stil (yazı tipi + kart düzeni + renkler).
#  Not: Tek bir stil bloğu; renkleri sakin bir "klinik" palete göre ayarlar.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2.2rem; max-width: 1150px; }
    h1, h2, h3 { letter-spacing: -0.015em; }

    /* Sekme çubuğunu biraz yumuşat */
    .stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 500; }

    /* Teşhis sonuç kartı: tek renk ailesinde yumuşak geçiş + hafif gölge */
    .big-card {
        padding: 30px; border-radius: 20px; text-align: center; color: #fff;
        margin-top: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.10);
    }
    .big-card h1 { font-size: 1.9rem; margin: 0 0 6px 0; font-weight: 700; }
    .big-card h2 { font-size: 1.2rem; margin: 0 0 10px 0; font-weight: 500; opacity: .95; }
    .big-card p  { margin: 0; opacity: .9; font-size: .95rem; }
    .risk-high { background: linear-gradient(135deg, #c23b4b, #8e2434); }
    .risk-low  { background: linear-gradient(135deg, #2f9e6f, #1e7a54); }

    /* Kritik soru vurgu kutusu */
    .warn-box {
        background: #fff8f1; border: 1px solid #f3d9c2; border-left: 5px solid #d9822b;
        padding: 20px 22px; border-radius: 14px;
    }
    .warn-box h3 { margin-top: 0; color: #a85c13; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def model_yukle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def veri_yukle():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


kayit = model_yukle()
df = veri_yukle()

st.title("❤️ Kalp Hastalığı Tahmini ve Klinik Karar Destek Arayüzü")
st.caption("Heart Disease UCI veri seti • Logistic Regression / Decision Tree / Random Forest")

if kayit is None:
    st.error(
        "⚠️ `model.pkl` bulunamadı. Lütfen önce terminalde şu komutu çalıştırın:\n\n"
        "`python train_and_analyze.py`"
    )
    st.stop()

model = kayit["model"]
scaler = kayit["scaler"]
features = kayit["features"]
column_tr = kayit["column_tr"]
sonuc_df = kayit["sonuc_df"]
hatali = kayit["hatali_ornekler"]

tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Canlı Teşhis Simülatörü",
    "📊 Veri Analizi & 5 Grafik",
    "🏆 Model Karşılaştırması",
    "🔍 Model Neden Yanıldı?",
])

# =========================================================================== #
#  SEKME 1 — CANLI KLİNİK TEŞHİS SİMÜLATÖRÜ
# =========================================================================== #
with tab1:
    st.subheader("🩺 Hasta Bilgilerini Girin")
    st.write("Soldaki değerleri ayarlayın; model anında kalp hastalığı riskini tahmin etsin.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Yaş", 20, 90, 54)
        sex = st.selectbox("Cinsiyet", [("Erkek", 1), ("Kadın", 0)],
                           format_func=lambda x: x[0])[1]
        cp = st.selectbox(
            "Göğüs Ağrısı Tipi",
            [("0 - Tipik Anjina", 0), ("1 - Atipik Anjina", 1),
             ("2 - Non-anjinal", 2), ("3 - Asemptomatik", 3)],
            format_func=lambda x: x[0])[1]
        trestbps = st.slider("Dinlenme Kan Basıncı (mm Hg)", 90, 200, 130)
        chol = st.slider("Kolesterol (mg/dl)", 120, 570, 246)
    with c2:
        fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl?", [("Hayır", 0), ("Evet", 1)],
                           format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Dinlenme EKG", [("0 - Normal", 0),
                               ("1 - ST-T Anormal", 1), ("2 - LV Hipertrofi", 2)],
                               format_func=lambda x: x[0])[1]
        thalach = st.slider("Maksimum Kalp Atış Hızı", 70, 210, 150)
        exang = st.selectbox("Egzersize Bağlı Anjina?", [("Hayır", 0), ("Evet", 1)],
                             format_func=lambda x: x[0])[1]
        oldpeak = st.slider("ST Depresyonu (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    with c3:
        slope = st.selectbox("ST Eğimi (slope)", [("0 - Yukarı", 0),
                             ("1 - Düz", 1), ("2 - Aşağı", 2)],
                             format_func=lambda x: x[0])[1]
        ca = st.selectbox("Boyanan Ana Damar Sayısı", [0, 1, 2, 3])
        thal = st.selectbox("Talasemi (thal)", [("1 - Normal", 1),
                            ("2 - Sabit Defekt", 2), ("3 - Geri Dönüşlü", 3)],
                            format_func=lambda x: x[0])[1]

    # Girdi vektörünü eğitimdeki sütun sırasına göre hazırla
    girdi = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    X = pd.DataFrame([[girdi[f] for f in features]], columns=features)
    X_s = scaler.transform(X)

    tahmin = int(model.predict(X_s)[0])
    try:
        olasilik = float(model.predict_proba(X_s)[0][1]) * 100
    except Exception:
        olasilik = 100.0 if tahmin == 1 else 0.0

    st.markdown("---")
    if tahmin == 1:
        st.markdown(
            f"<div class='big-card risk-high'>"
            f"<h1>⚠️ EVET — Yüksek Kalp Hastalığı Riski</h1>"
            f"<h2>Risk Olasılığı: %{olasilik:.1f}</h2>"
            f"<p>Model bu hastada kalp hastalığı öngörüyor. Klinik değerlendirme önerilir.</p>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='big-card risk-low'>"
            f"<h1>✅ HAYIR — Düşük Risk</h1>"
            f"<h2>Risk Olasılığı: %{olasilik:.1f}</h2>"
            f"<p>Model bu hastada kalp hastalığı öngörmüyor. Yine de rutin kontroller önemlidir.</p>"
            f"</div>", unsafe_allow_html=True)

    st.progress(min(max(olasilik / 100, 0.0), 1.0))
    st.caption(f"Tahmini yapan model: **{kayit['model_adi']}**")

# =========================================================================== #
#  SEKME 2 — VERİ ANALİZİ & 5 GRAFİK
# =========================================================================== #
with tab2:
    st.subheader("📊 Veri Setinin Özeti")

    if df is not None:
        n = len(df)
        n_evet = int((df["target"] == 1).sum())
        n_hayir = int((df["target"] == 0).sum())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Toplam Kayıt", n)
        m2.metric("EVET (Hasta)", n_evet, f"%{100*n_evet/n:.0f}")
        m3.metric("HAYIR (Sağlıklı)", n_hayir, f"%{100*n_hayir/n:.0f}")
        m4.metric("Özellik Sayısı", len(features))

        with st.expander("🔎 Sütunların Türkçe Açıklamaları"):
            st.table(pd.DataFrame(
                [(k, v) for k, v in column_tr.items()],
                columns=["Sütun", "Açıklama"]))

        with st.expander("📋 Ham Verinin İlk 10 Satırı"):
            st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("5 Analiz Grafiği ve Yorumları")

    grafikler = [
        ("1_sinif_dagilimi.png", "Grafik 1 — Sınıf Dağılımı",
         "Veri setinde 'Evet' ve 'Hayır' sınıfları birbirine yakındır; "
         "yani veri makul derecede **dengelidir**. Bu, modelin tek bir sınıfa "
         "yanlı (bias) olmadan öğrenebilmesi için önemlidir."),
        ("2_cinsiyet_hastalik.png", "Grafik 2 — Cinsiyete Göre Kalp Hastalığı",
         "Erkek hastalarda kalp hastalığı görülme sayısı belirgin şekilde daha "
         "yüksektir. Cinsiyet, modelin dikkate aldığı anlamlı bir risk faktörüdür."),
        ("3_gogus_agrisi_etkisi.png", "Grafik 3 — Göğüs Ağrısı Tipine Göre Hastalık",
         "İlginç biçimde **asemptomatik (tip 0)** göğüs ağrısı olan hastalarda "
         "hastalık oranı yüksektir. Bu, 'ağrım yok' diyen hastaların bile riskli "
         "olabileceğini, yani klinik dikkatin önemini gösterir."),
        ("4_yas_ve_maks_nabiz.png", "Grafik 4 — Yaş ve Maksimum Kalp Atış Hızı",
         "Maksimum kalp atış hızı (thalach) **düşük** olan ve yaşı **ileri** olan "
         "hastalar daha çok hasta sınıfında toplanır. Yüksek nabza ulaşabilen "
         "genç hastalar genelde sağlıklı tarafta yer alır."),
        ("5_korelasyon_haritasi.png", "Grafik 5 — Korelasyon Isı Haritası",
         "Hedef ile en güçlü ilişkiye sahip değişkenler **cp (göğüs ağrısı), "
         "thalach (maks. nabız), oldpeak, exang ve ca**'dır. Bu değişkenler "
         "modelin tahminlerinde en etkili olanlardır."),
    ]

    for i in range(0, len(grafikler), 2):
        cols = st.columns(2)
        for col, (dosya, baslik, yorum) in zip(cols, grafikler[i:i + 2]):
            with col:
                yol = os.path.join(PLOTS_DIR, dosya)
                st.markdown(f"**{baslik}**")
                if os.path.exists(yol):
                    st.image(yol, use_container_width=True)
                else:
                    st.warning(f"{dosya} bulunamadı. Önce train_and_analyze.py çalıştırın.")
                st.info(yorum)

# =========================================================================== #
#  SEKME 3 — MODEL KARŞILAŞTIRMASI & CONFUSION MATRIX
# =========================================================================== #
with tab3:
    st.subheader("🏆 3 Modelin Karşılaştırması")
    st.dataframe(
        sonuc_df.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}",
            "Recall": "{:.3f}", "F1-Score": "{:.3f}",
        }).highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score"],
                         color="#d4efdf"),
        use_container_width=True,
    )
    st.caption(f"Seçilen en iyi model (Recall önceliğiyle): **{kayit['model_adi']}**")

    c1, c2 = st.columns(2)
    with c1:
        yol = os.path.join(PLOTS_DIR, "6_model_karsilastirma.png")
        if os.path.exists(yol):
            st.image(yol, use_container_width=True)
    with c2:
        yol = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        if os.path.exists(yol):
            st.image(yol, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<div class='warn-box'>"
        "<h3>🚨 Neden Yanlış 'HAYIR' Demek Daha Kötüdür?</h3>"
        "<b>Yanlış 'HAYIR' (False Negative):</b> Kalp hastası olan birine "
        "'sağlıklısın' demektir. Hasta tedavisiz kalır → <b>hayati risk, ölüm.</b><br><br>"
        "<b>Yanlış 'EVET' (False Positive):</b> Sağlıklı birine 'hastasın' demektir. "
        "Sonucu sadece <b>ek bir doktor kontrolü / test maliyeti</b>dir.<br><br>"
        "➡️ Bu yüzden sağlıkta <b>Recall (Duyarlılık)</b> metriği en kritik olandır: "
        "gerçek hastaları kaçırmamak önceliğimizdir."
        "</div>",
        unsafe_allow_html=True,
    )

# =========================================================================== #
#  SEKME 4 — MODEL NEDEN YANILDI? (HATA ANALİZİ)
# =========================================================================== #
with tab4:
    st.subheader("🔍 Model Neden Yanıldı? — Hata Analizi")
    st.write(
        "Aşağıda modelin **test setinde yanlış tahmin ettiği** gerçek hasta "
        "örnekleri ve her birinin neden yanlış tahmin edildiğine dair yorum yer alır."
    )

    if hatali is not None and len(hatali) > 0:
        goster = hatali.head(4).copy()
        for _, satir in goster.iterrows():
            gercek = "EVET (Hasta)" if satir["Gercek"] == 1 else "HAYIR (Sağlıklı)"
            tahmin_l = "EVET (Hasta)" if satir["Tahmin"] == 1 else "HAYIR (Sağlıklı)"
            fn = satir["Gercek"] == 1 and satir["Tahmin"] == 0
            with st.container():
                st.markdown(
                    f"**Hasta:** {int(satir['age'])} yaşında • "
                    f"Gerçek: `{gercek}` → Model: `{tahmin_l}` "
                    f"{'🔴 (TEHLİKELİ FN)' if fn else '🟡 (FP)'}"
                )
                if fn:
                    st.error(satir["Yorum"])
                else:
                    st.warning(satir["Yorum"])
                st.markdown("---")

        st.info(
            "**Özet:** Kırmızı (FN) örnekler en tehlikelileridir — model gerçek bir "
            "hastayı 'sağlıklı' sandı. Sarı (FP) örneklerde ise sağlıklı birine "
            "gereksiz 'hasta' teşhisi konmuş, ki bu daha az risklidir."
        )

        with st.expander("📊 Yanlış tahmin edilen tüm örnekler (tablo)"):
            st.dataframe(
                hatali[["age", "sex", "cp", "chol", "thalach", "oldpeak",
                        "Gercek", "Tahmin"]],
                use_container_width=True,
            )
    else:
        st.success("Bu çalışmada test setinde hiç yanlış tahmin bulunmadı. 🎉")
