# CMP4509 teslim kontrol listesi

Bu belge ders teslimlerini ve deneylerden toplanacak kanıtları izler. Tarihler Türkiye saatine göredir. Paper sign-up tamamlandı.

| Teslim | Tarih / şart | Şimdiki durum |
|---|---|---|
| Public GitHub deposu | Kaynak + tekrar üretim talimatları; raporda bağlantısı olmalı | [Yayımlandı: emirbakac/cmp4509-madry-mnist](https://github.com/emirbakac/cmp4509-madry-mnist) |
| Bağımsız notebook | Temiz Colab/Kaggle oturumunda yukarıdan aşağı çalışmalı | Varsayılan smoke; gerçek GPU sonuçları kullanıcı tarafından çalıştırılacak |
| Sunum | 30 Kasım **veya** 7 Aralık 2026; QA dahil 10 dakika | Grubun kesin tarihini ders duyurusundan netleştir |
| Rapor PDF | **13 Aralık 2026, 23:59**; Teams üzerinden | Şablon ve bilimsel deney sonuçları bekleniyor |
| Ders şablonu | Teams: `CMP4509_report_template.zip` | Bu çalışma alanında henüz bulunmuyor |

## Önerilen sıra

1. **İlk çalışma oturumu:** smoke test; kod ve saldırı mekanizmasını birlikte anlama; çalışan notebook ekran görüntüsü.
2. **Sonraki GPU oturumu:** pilot normal eğitim ve pilot PGD eğitimi; süre ölçümü; doğrulama eğrileri; hata notları.
3. **Ana deney öncesi:** ayarları ve seed listesini dondur; grup üyelerine koşuları dağıt; kalıcı kayıt yerini doğrula.
4. **Ana deney:** her iki model için üç seed; bütün checkpoint, config, geçmiş ve ortam dosyalarını sakla.
5. **Güçlü test:** aynı PGD 100 adım × 20 restart protokolü; gerçek CSV ve şekiller; tabloyu bağımsız olarak kontrol et.
6. **Sunumdan önce:** 8–10 slayt; yaklaşık 7–8 dakika anlatım + 2–3 dakika sorular için prova. Her üye tüm akışı açıklasın.
7. **Rapor:** gerçek ders LaTeX şablonunu kullan; public depo bağlantısını ekle; PDF'yi görsel olarak kontrol et; Teams teslimini doğrula.

## Public GitHub deposu

Depo görünürlüğü gerçekten **public** olmalı; bağlantıyı oturum açılmamış bir pencerede açarak doğrula. Şunlar depoda yer alsın:

- README ve çalıştırma sırası.
- Bütün kaynak kodu, requirements ve isimlendirilmiş config dosyaları.
- Kaynaklarını içeren notebook; tek bir Colab oturumuna veya özel Drive yoluna bağımlı olmamalı.
- Gerçek sonuç tablosu/şekilleri, kullanılan seed'ler ve saldırı bütçesi.
- Makale ve resmî yazar uygulamasına atıf; birebir tekrar olmayan noktaların açıklaması.

MNIST verisi, büyük checkpoint'ler, sanal ortamlar ve erişim anahtarları repoya eklenmesin. Checkpoint'ler için paylaşılabilir ayrı bir çıktı bağlantısı kullanılabilir. Public depo açıldığında README'deki yayın bağlantısını ve rapordaki URL'yi güncelle; bir repo URL'sini var olmadan yazma.

## Rapor iskeleti: şablona aktarılacak içerik

Bu bir içerik planıdır; dersin biçim şablonu yerine geçmez.

1. Problem ve motivasyon: temiz doğruluk neden tek başına yeterli değil?
2. Madry çalışması: min–max hedefi, L∞ tehdit modeli, PGD ile iç optimizasyon.
3. Yöntem: veri bölümü, iki convolution ağı, baseline/PGD eğitimi, optimizer, bütçe, seed ve checkpoint seçimi.
4. Değerlendirme: temiz/FGSM/PGD, rastgele başlangıçlar, restart başarısızlığını birleştirme, testin dondurulması.
5. Gerçek sonuçlar: seed başına tablo, ortalama ± std, süre, grafikler ve örnekler.
6. Tartışma: temiz doğruluk–dayanıklılık ilişkisi, güçlü testin etkisi, makaleden farklar ve sınırlılıklar.
7. Sonuç, katkı dağılımı, kaynakça ve public depo bağlantısı.

Sonuç gelmeden sayı, “başarılı tekrar” iddiası veya nihai sonuç paragrafı doldurma. Makaleden alınan sayı varsa açıkça **literatür sonucu** olarak etiketle ve kendi tablonla karıştırma.

## Sunumda herkesin açıklayabilmesi gerekenler

- İçteki max ve dıştaki min ne yapıyor?
- ε=0.3 hangi piksel ölçeğinde? L∞ sınırı ne demek?
- FGSM ile PGD arasındaki fark ne?
- Rastgele başlangıç/restart neden gerekli?
- Eğitimden daha güçlü test neden kullanılıyor?
- Hangi ayarlar makaleden farklı ve neden?
- Veri sızıntısı olmadan checkpoint nasıl seçildi?
- Sonuç üç seed'de ne kadar değişti? Hesaplama maliyeti neydi?

## Ekran görüntüsü ve çıktı listesi

- [ ] Ortam/GPU ve config hücresi.
- [ ] Smoke testin tamamlanması.
- [ ] Baseline ve PGD eğitim eğrileri.
- [ ] Aynı görüntünün temiz/FGSM/PGD örneği ve ε kontrolü.
- [ ] Seed başına gerçek sonuç CSV tablosu.
- [ ] Ortalama ± std karşılaştırma grafiği.
- [ ] Public GitHub ana sayfası ve notebook.
- [ ] Doğru şablonla üretilmiş son PDF.
- [ ] Teams teslim onayı.

Ekran görüntüsü sayısal dosyaların yerine geçmez. CSV, config ve kaynak sürümü temel kayıttır; görüntüler anlatımı destekler. Bu belge bir kontrol listesidir; otomatik hatırlatıcı kurulmuş değildir.
