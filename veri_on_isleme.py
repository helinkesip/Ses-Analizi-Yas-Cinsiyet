import pandas as pd
import os

# Dosya yolunu tanımlayalım 
dosya_yolu = 'tr/validated.tsv' 

if os.path.exists(dosya_yolu):
    df = pd.read_csv(dosya_yolu, sep='\t', low_memory=False)
    print(f"Dosya başarıyla bulundu! Toplam satır: {len(df)}")
    
    # Cinsiyet ve Yaş dolu olanları filtrele
    df_temiz = df.dropna(subset=['gender', 'age'])
    
    # Sadece gerekli sütunları ve ilk 5000 örneği alalım 
    df_final = df_temiz[['path', 'age', 'gender']]
    
    print(f"Filtrelenmiş (Etiketli) kayıt sayısı: {len(df_final)}")
    df_final.to_csv('temiz_veri_listesi.csv', index=False)
    print("'temiz_veri_listesi.csv' dosyası oluşturuldu.")
else:
    print(f"Hata: {dosya_yolu} bulunamadı! Lütfen klasör ismini kontrol et.")
