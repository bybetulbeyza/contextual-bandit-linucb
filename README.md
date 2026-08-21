# Contextual Bandit - LinUCB

Kullanıcı etkileşimlerini ve bağlamsal özelliklerini (context) dikkate alarak dinamik ve kişiselleştirilmiş karar verme süreçlerini optimize eden **LinUCB (Linear Upper Confidence Bound)** tabanlı pekiştirmeli öğrenme projesi.

---

## Overview

* **Amaç:** Geleneksel A/B testlerinin getirdiği keşif maliyetini azaltarak, kullanıcı özelliklerine göre en yüksek geri dönüşü (CTR / Reward) sağlayacak seçeneği gerçek zamanlı olarak belirlemek.
* **LinUCB Rolü:** Exploration (keşfetme) ve Exploitation (eldeki bilgiyi kullanma) dengesini kurarak, doğrusal regresyon yaklaşımıyla her bir seçenek için beklenen ödülü ve belirsizliği hesaplar.
* **Projedeki Bileşenler:**
  * **Context ($x$):** Kullanıcının demografik, davranışsal ve anlık durum özellikleri (yaş, cihaz, bölge, harcama geçmişi vb.).
  * **Arm ($a$):** Kullanıcıya sunulabilecek farklı seçenekler/içerikler (Örn: 5 farklı kampanya/reklam alternatifi — `Arm 0` - `Arm 4`).
  * **Reward ($r$):** Kullanıcının gösterilen içeriğe verdiği pozitif yanıt / tıklama (Binary: `1` veya `0`).

---

## Dataset

Veri seti, platformdaki kullanıcıların özelliklerini ve her bir arm'a karşı gösterdikleri gerçek/potansiyel tıklama yanıtlarını içerir:

* **Context Özellikleri:**
  * `age`: Kullanıcı yaşı
  * `gender`: Cinsiyet (`F`, `M`)
  * `device`: Kullanılan cihaz (`mobile`, `desktop`, `tablet`)
  * `region`: Coğrafi bölge (`marmara`, `ege`, `akdeniz`, `ic_anadolu`, vb.)
  * `time_of_day`: Günün zamanı (`morning`, `afternoon`, `evening`, `night`)
  * `session_count`: Oturum sıklığı
  * `avg_purchase`: Geçmiş ortalama alışveriş tutarı
* **Reward / Arm Yapısı:**
  * `reward_arm_0` ... `reward_arm_4`: Her bir arm seçildiğinde elde edilecek ikili (binary) ödül değeri.

---

## Method (LinUCB)

LinUCB algoritması her adımda şu döngüyü takip eder:

1. **Context Bilgisi:** Kullanıcının normalize edilmiş özellik vektörü ($x_t$) alınır.
2. **Tahmin & Belirsizlik:** Her bir arm için doğrusal model üzerinden tahmini getiri ve kovaryans matrisi üzerinden belirsizlik (uncertainty) hesaplanır.
3. **Arm Seçimi (UCB Skoru):** 
   $$	ext{Score}_a = 	ext{Prediction}_a + lpha 	imes 	ext{Uncertainty}_a$$
   formülü ile en yüksek skora sahip arm seçilir.
4. **Geri Bildirim (Reward):** Seçilen arm için gerçek ödül ($r_t \in \{0, 1\}$) gözlemlenir.
5. **Model Güncelleme:** İlgili arm'a ait $A_a$ matrisi ve $b_a$ vektörü güncellenerek model bir sonraki kullanıcı için daha optimize hale getirilir.

---

## Preprocessing

* **Kategorik Değişkenler:** `gender`, `device`, `region`, `time_of_day` → **One-Hot Encoding**
* **Sayısal Değişkenler:** `age`, `session_count`, `avg_purchase` → **StandardScaler**
* **Reward Ayrımı:** `reward_arm_0` – `reward_arm_4` hedef sütunları context matrisinden ayrı tutularak simülasyon ortamında geri bildirim mekanizması olarak kullanılmıştır.

---

## Technologies

* **Python 3.x**
* **NumPy** – Matris işlemleri ve UCB skor hesaplamaları
* **Pandas** – Veri analizi ve manipülasyonu
* **Scikit-learn** – Veri ön işleme (One-Hot Encoding & Scaling)

---

## Results

| Arm | Seçilme Sayısı | Toplam Tıklama (Reward) | Başarı Oranı (CTR) |
| :--- | :---: | :---: | :---: |
| **Arm 0** | ~1,850 | ~820 | **%44.3** |
| **Arm 1** | ~1,420 | ~540 | **%38.0** |
| **Arm 2** | ~1,260 | ~390 | **%31.0** |
| **Arm 3** | ~1,410 | ~470 | **%33.3** |
| **Arm 4** | ~1,340 | ~410 | **%30.6** |

> **Yorum:** LinUCB algoritması ilk adımlardaki keşif (exploration) sürecinin ardından kullanıcı context'ine göre en yüksek dönüşüm getiren kolları (özellikle Arm 0 ve Arm 1) ağırlıklı olarak seçmeyi öğrenmiş ve kümülatif ödülü belirgin şekilde artırmıştır.

---

## Project Structure

```text
contextual-bandit-linucb/
├── contextual_bandit_dataset.csv   # Kullanıcı bağlam ve ödül veri seti
├── linucb_model.py                 # Ön işleme ve LinUCB simülasyon kodu
├── requirements.txt                # Gerekli kütüphaneler
└── README.md                       # Proje dokümantasyonu
```