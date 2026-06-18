import pandas as pd
import librosa
import numpy as np
import os

# 1. Temiz listeyi yükle
df = pd.read_csv('temiz_veri_listesi.csv')


df = df.head(1000)

def feature_extractor(file_path):
    # Ses dosyasını yükle 
    full_path = os.path.join('tr', 'clips', file_path)
    
    try:
        # Sesi yükle ve 22kHz örnekleme hızına sabitle
        audio, sample_rate = librosa.load(full_path, sr=None) # Orijinal örnekleme hızında yükle
        
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
features = []
genders = []
ages = []
for index, row in df.iterrows():
    data = feature_extractor(row['path'])
    if data is not None:
        features.append(data)
        genders.append(row['gender'])
        ages.append(row['age'])

# Numpy dizilerine dönüştür
X = np.array(features)
genders = np.array(genders)
ages = np.array(ages)

# Veriyi bir DataFrame'e dönüştür ve kaydet
features_df = pd.DataFrame({
    'feature': list(X),
    'gender': genders,
    'age': ages
})

np.savez('ozellikler.npz', X=X, gender=genders, age=ages)

print(f"Başarıyla {len(features_df)} sesin özelliği çıkarıldı ve 'ozellikler.npz' olarak kaydedildi.")
