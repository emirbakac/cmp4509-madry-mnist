# CMP4509 teslim durumu ve son kontrol listesi

Tarihler Türkiye saatine göredir. Paper sign-up tamamlandı; proje dokuz modelin final-test ölçümünü içeriyor. Eğitim sonuçlarının artık var olması, bütün teslim koşullarının tamamlandığı anlamına gelmez.

| Teslim | Şart | Kanıt / kalan adım |
|---|---|---|
| Public GitHub | Kaynak, notebook, yeniden üretim adımları ve raporda bağlantı | [Public depo](https://github.com/emirbakac/cmp4509-madry-mnist) mevcut; audit branch/PR durumu FINAL_AUDIT'te |
| Bağımsız eğitim notebook'u | Kaynak/config dosyalarını içermeli; smoke ve final100×5 akışı | `notebooks/01_mnist_reproduction.ipynb`; güncel çalıştırma kanıtı VALIDATION/FINAL_AUDIT'te |
| Sonuç analizi | Özgün dokuz kayıt korunmalı; ortalama ± örnek std | `results/FINAL_TEST_ALL_MODELS.csv`, provenance JSON ve yeniden hesaplanan çıktılar |
| Rapor PDF | **13 Aralık 2026, 23:59**, Teams; gerçek ders LaTeX şablonu | `report/` içinde teslim edilen kaynaklar mevcut; derleme/inceleme sonucu FINAL_AUDIT'te |
| Resmî şablon kökeni | Teams'teki `CMP4509_report_template.zip` ile doğrulama | Özgün şablon arşivi bulunmadığı sürece **teslim engeli**; mevcut kaynakların benzer görünmesi kanıt değil |
| Sunum | 30 Kasım veya 7 Aralık 2026; QA dahil 10 dakika | Ayrı aşama; bu final audit'te slayt üretilmiyor |

## Teslimden önce

1. Özgün ders şablonunu temin et ve rapor biçimiyle karşılaştır. Alternatif bir tasarımı sessizce ders şablonu diye kullanma.
2. Son raporun, kontrol edilen `report/main.tex` ve kaynakçadan üretildiğini doğrula. Kılavuz kutuları kapalı, denklemler/citations çözülmüş, grafikler okunaklı olmalı.
3. Final tabloların **10.000 test görüntüsü, PGD100×5**; ek güçlü kontrollerin **512 validation görüntüsü, PGD100×20/AutoAttack** olduğunu ayrı belirt.
4. PGD40 seed1/2 başarısızlıklarını ve curriculum'un sonradan geliştirilen müdahale olduğunu koru. Tarihî checkpoint transferi ile entegre sıfırdan çalıştırmayı eşdeğer dosyalar gibi sunma.
5. Grup üyeleri Contributions ve Use of AI Tools metinlerini doğrulasın. Atanan sorumluluklar, gerçekten tamamlandığı doğrulanan işler gibi yazılmamalı.
6. GitHub'da gerekli kaynak/config/notebook/sonuç/rapor dosyalarının bulunduğunu kontrol et; audit değişiklikleri gözden geçirilmeden otomatik merge yapılmaz.
7. PDF'yi Teams'e yükle ve teslim onayını sakla. Bu belge otomatik hatırlatıcı veya teslim işlemi değildir.

## Saklanması yararlı kanıtlar

- Cihaz, ortam ve config ekran görüntüsü.
- Smoke notebook'un gerçek tamamlanma çıktısı.
- Seed başına özgün CSV, saldırı bütçesi, seçilen epoch, checkpoint hash'i.
- Varsa tam eğitim geçmişleri ve örnek tahminler; yoksa öğrenme eğrisi üretme.
- Ortalama ± sample SD figürleri ve doğrulanmış veri kaynağı.
- Validation saldırılarının ham logları, örnek indeksleri, AutoAttack sürümü ve Square uyarısı.
- Public repo, son PDF ve Teams teslim onayı.

CSV/config/log dosyaları temel kayıttır. Ekran görüntüleri destekleyicidir. Büyük checkpoint'ler ve MNIST verisini Git'e koymak yerine erişilebilir ayrı bir arşivde sakla; erişim anahtarlarını veya kişisel dosyaları yayımlama.
