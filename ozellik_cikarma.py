import pandas as pd
import librosa
import numpy as np
import os

# 1. Temiz listeyi yükle
df = pd.read_csv('temiz_veri_listesi.csv')

# İlk deneme için sadece 1000 tane veriyle çalışalım (Hız için)
# Her şey yolunda giderse bu satırı silebiliriz
df = df.head(1000)

def feature_extractor(file_path):
    # Ses dosyasını yükle (clips klasörünün içinde olduğunu belirtiyoruz)
    # tr/clips/ yolunu kontrol et, sende farklı olabilir
    full_path = os.path.join('tr', 'clips', file_path)
    
    try:
        # Sesi yükle ve 22kHz örnekleme hızına sabitle
        audio, sample_rate = librosa.load(full_path, res_type='kaiser_fast')
        
        # MFCC özelliklerini çıkar (Genelde 40 adet katsayı yeterlidir)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        
        # Özelliklerin ortalamasını alarak vektöre dönüştür
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        
        return mfccs_scaled_features
    except Exception as e:
        print(f"Hata: {file_path} okunamadı. {e}")
        return None

print("Ses özellikleri çıkarılıyor, bu işlem biraz sürebilir...")

# Özellikleri listeye ekle
extracted_features = []
for index, row in df.iterrows():
    data = feature_extractor(row['path'])
    if data is not None:
        label_gender = row['gender']
        label_age = row['age']
        extracted_features.append([data, label_gender, label_age])

# Veriyi bir DataFrame'e dönüştür ve kaydet
features_df = pd.DataFrame(extracted_features, columns=['feature', 'gender', 'age'])
np.save('ozellikler.npy', extracted_features) # Veriyi binary olarak kaydetmek daha hızlıdır

print(f"Başarıyla {len(features_df)} sesin özelliği çıkarıldı ve 'ozellikler.npy' olarak kaydedildi.")
