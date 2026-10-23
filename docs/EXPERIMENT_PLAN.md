# Deney planı: önce çalışan zincir, sonra anlamlı karşılaştırma

## 1. Smoke test: bağlantılar doğru mu?

Notebook'un varsayılan `smoke` ayarını iki eğitim türünde çalıştır. Bu koşuda amaç iyi doğruluk almak değildir. Şunları görmeliyiz:

- MNIST indiriliyor ve tensör değerleri `[0,1]` aralığında.
- İki convolution katmanlı ağ ileri/geri geçiş yapıyor.
- FGSM ve PGD görüntüyü `[0,1]` aralığında ve orijinale L∞ uzaklığı `ε≤0.3` olacak şekilde tutuyor.
- Eğitim kaybı, doğrulama, checkpoint, CSV ve görseller üretiliyor.
- Daha fazla restart altında başarıyla saldırılmış bir örnek yanlışlıkla tekrar doğru sayılmıyor.

**Ekran görüntüsü:** GPU/CPU bilgisi, tüm smoke hücrelerinin tamamlandığı ekran, temiz ve saldırılmış görüntü örnekleri. Bunlar “deney ortamı çalışıyor” kanıtıdır; asıl başarı sonucu değildir.

## 2. Pilot: öğrenme ve süreyi gözle

`pilot`, önce `baseline`, sonra `adversarial`; başlangıçta seed 0. `--train-only` kullan. Normal eğitim sadece temiz görüntüyü görür. Adversarial eğitim her batch için mevcut modelle PGD örnekleri üretir ve o örneklerde güncelleme yapar.

Aynı veri bölümü ve aynı ağ, iki eğitim yönteminin karşılaştırılmasını kolaylaştırır. Eşit epoch sayısı eşit hesaplama maliyeti değildir: PGD eğitimi birçok ek gradyan hesabı yapar. Süreyi de raporlamak bu farkı görünür kılar.

Kontrol et: kayıp sonlu mu, temiz doğrulama öğreniyor mu, PGD doğrulama nasıl değişiyor, bir epoch ne kadar sürüyor? Hiperparametre kararlarını burada ve doğrulama kümesinde ver. Test setini model seçimi için kullanma.

**Ekran görüntüsü:** iki eğitim türünün öğrenme eğrileri, kullanılan config ve süre. Bir hata varsa hata mesajını ve çözümünü deney günlüğüne yaz.

## 3. Ana deney: karşılaştırmayı önceden sabitle

Pilot sonrası ağ genişliği, epoch, optimizer, öğrenme oranı, eğitim PGD bütçesi ve checkpoint seçim kuralını dondur. Varsayılan `main` ayarları nihai karar yerine ilk bütçe önerisidir. Değişiklikleri yeni config dosyasında sakla.

Her iki eğitim yöntemi için seed kümesi **[0,1,2]**. Aynı seed numaralarını eşleştir ve bütün koşuları sakla. Eğitim/doğrulama bölümü sabit kalır; seed ağırlık başlangıcı, batch sırası ve saldırı rastgeleliğini etkiler. Donanım veya yazılım değişimleri tam bit düzeyinde eşitliği etkileyebilir; ortam kaydını karşılaştır.

Normal eğitim checkpoint'i temiz doğrulama doğruluğuyla; adversarial eğitim checkpoint'i saldırılı doğrulama doğruluğuyla seçilir. Bu farklılık raporda açık olmalı. Eğitim sonunda `last.pt` mevcut olsa da test için önceden belirlenen `best.pt` kullanılır.

Uzun koşuda saklama konumunu önce ayarla. Colab için Drive, Kaggle için `/kaggle/working` ve Save Version kullan. Bir epoch bitmeden oturum kesilirse o epoch yeniden çalışabilir.

## 4. Dondurulmuş test: saldırıyı güçlendir

Bütün kararlar verildikten sonra her checkpoint için:

| Ölçüm | Amaç |
|---|---|
| Temiz test doğruluğu | Normal görüntülerde performans |
| FGSM, ε=0.3 | Tek adımlı saldırıyla karşılaştırma |
| PGD, ε=0.3, α=0.01, 100 adım, 20 restart | Eğitimden güçlü beyaz-kutu dayanıklılık testi |
| Koşu ve değerlendirme süresi | Ücretsiz GPU bütçesini görünür kılma |

PGD için modelin gradyanı kullanılabilir; saldırı **beyaz-kutu** ve **hedefsizdir**. Her restart kaybı büyütmeye çalışır. Örneğin herhangi bir başlangıç/adım/restart ile yanlış sınıflandırılması başarısızlık sayılır. Sıfır değişim de tehdit kümesinde olduğundan temizken yanlış olan örnek dayanıklı kabul edilmez.

Finalden önce küçük doğrulama altkümesinde 40/100 adım ve 1/10/20 restart bütçelerini karşılaştırabilirsin. Bu bir saldırı kontrolüdür; test sonuçlarına göre ağı yeniden ayarlama. Birden çok saldırı bütçesini raporlarsan hangi checkpoint ve örnek kümesini kullandığını açık yaz. Daha güçlü saldırı bütçesinin tek rastgele çalışmada mutlaka daha düşük sayı üretmesi garanti değildir; ortak başlangıçlar ve birikimli başarısızlık takibi bu kontrolü daha anlamlı kılar.

GPU süresi yetmezse sonucu gizleme: daha küçük bütçeyi açıkça yaz, bütün modellerde aynı protokolü kullan ve güçlü testin eksik kaldığını sınırlılık olarak belirt. Tam veri testi yapılmış gibi konuşma.

## 5. Birleştirme ve yorum

Özet CSV'de seed başına temiz/FGSM/PGD doğrulukları; toplu tabloda model türü, seed sayısı, ortalama ve **örnek standart sapma (ddof=1)** yer alır. Tek seed için standart sapma sıfırmış gibi sunulmaz. Yüzde değer ile yüzde puan farkını ayır: iki modelin doğruluk farkı yüzde puandır.

Farklı config, veri kaynağı, test örnek sayısı, saldırı adımı veya restart bütçelerini tek ortalamada karıştırma. `smoke` ve `pilot` ana deneyin satır sayısını artırmaz. Üç seed belirsizliği göstermek için mütevazı bir başlangıçtır; güçlü istatistiksel genelleme iddiası kurma.

Raporun merkezindeki grafik aynı protokol altında **normal eğitim / PGD eğitimi × temiz / FGSM / güçlü PGD** karşılaştırmasıdır. İkinci şekil öğrenme eğrileri; üçüncü şekil birkaç aynı örneğin temiz/saldırılmış görünümü olabilir. Örnekler tek başına nicel sonuç yerine geçmez.

Şu sorulara sonuçlardan cevap ver:

1. PGD eğitimi güçlü saldırı altındaki doğruluğu artırdı mı?
2. Temiz doğrulukta ve hesaplama süresinde ne değişti?
3. Sonuç seed'ler arasında tutarlı mı?
4. Daha güçlü test önceki izlenimimizi değiştirdi mi?
5. Makaledeki kurulumdan hangi farklar sonuçları açıklayabilir?

Beklenen yön bir hipotezdir. Sonuç beklediğimiz gibi çıkmazsa kod, saldırı bütçesi ve optimizasyonu inceleyip eldeki sayıları dürüstçe raporlarız.

## Deney günlüğü şablonu

Her önemli koşu için bu alanları bir Markdown dosyasına veya paylaşılan grup günlüğüne ekle:

```text
Tarih / çalıştıran kişi:
Git commit veya kullanılan kaynak arşivi:
Config adı ve değişiklikler:
Seed / model türü:
GPU ve ortam:
Checkpoint yolu:
Epoch / eğitim süresi:
Doğrulama gözlemi:
Test açıldı mı? Ayarlar donduruldu mu?
Saldırı bütçesi / test örnek sayısı:
Çıktı arşivi ve ekran görüntüsü:
Sonraki karar ve gerekçesi:
```
