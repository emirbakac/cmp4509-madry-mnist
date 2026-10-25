# Deney protokolü ve yeniden üretim rehberi

Proje artık bir başlangıç taslağı değil. `results/FINAL_TEST_ALL_MODELS.csv`, üç yöntem × üç seed için dokuz kayıt içeriyor. Bu belge mevcut sonuçların kökenini yeni yeniden üretim koşularından ayırır.

## Korunacak deney ayrımları

| Deney | Veri | Bütçe / amaç | Kanıt durumu |
|---|---|---|---|
| Ana final test | 10.000 MNIST test görüntüsü | PGD100×5, ε=.3, α=.01; clean/FGSM de ölçüldü | Dokuz satırlık özgün CSV mevcut; ortalamalar yeniden hesaplanabilir |
| Güçlü validation | Aynı olduğu bildirilen 512 validation görüntüsü | PGD100×20 ve full standard AutoAttack | Yazarların bildirdiği özet değerler; ham log/indeksler olmadan yeniden doğrulanmış sayılmaz |
| Rutin checkpoint seçimi | Sabit 5.000 validation görüntüsü | Her epoch PGD40×1; baseline için temiz doğruluk | Güncel kaynak kodunda incelenebilir |
| Smoke | 256/64/64 eğitim/validation/test | Küçük yazılım doğrulaması | Ana bilimsel sonuçlarla birleştirilmez |

Ana CSV'de saldırı bütçesi sütunları ve örnek bazında tahminler yoktur. PGD100×5 protokol tanımı tarihî deney açıklamasına dayanır; CSV kontrolü bu saldırı çalışmasının bağımsız tekrarı değildir. Kaydedilmiş SHA-256 değerlerini doğrulamak için özgün checkpoint dosyaları gerekir.

## Üç yöntem ve tarihî curriculum

1. **Baseline:** 10 epoch temiz örneklerle eğitim.
2. **PGD40 from scratch:** 10 epoch, ilk batch'ten itibaren PGD40. Başarısız seed1/2 çıkarılmaz: CSV'de clean/FGSM/PGD=.1135, seçilen epoch1. Tek sınıfa çökme ve 10 epoch'un tamamlanması yazarların süreç açıklamasıdır; CSV tek başına tahmin dağılımını veya bütün geçmişi kanıtlamaz. Ölü ReLU gibi nedensel açıklamalar hipotez olarak kalır.
3. **Curriculum:** epoch1–3 PGD10, epoch4–10 PGD40. Tarihî koşularda önce üç epoch'luk probe checkpoint'leri ayrı curriculum klasörlerine taşınarak model, Adam, RNG ve epoch durumunun korunduğu bildiriliyor. Güncel entegre program aynı 3+7 saldırı bütçesini sıfırdan uygular; iki yürütme yolunun bit düzeyinde eşdeğer olduğu gösterilmedi.

`main_pgd10_probe` araştırma tarihini belgeleyen ek config'tir. Entegre curriculum koşusu için önce probe eğitmek **gerekmez**. Sadece `--resume` kullanarak farklı config'li probe checkpoint'ini yeni koşuya aktarmak desteklenen bir yöntem değildir; uyumsuzluk denetimini kaldırma.

## Yeniden çalıştırma sırası

1. Notebook'u temiz oturumda `smoke` ile çalıştır. Cihaz, config ve tamamlanma ekranını sakla.
2. Gerekirse `pilot` ile süreyi ölç. Test için değil, kaynak bütçesi ve doğrulama davranışı için kullan.
3. `main` için baseline ve adversarial; `main_curriculum_pgd10_to_pgd40` için adversarial; her birinde seed0/1/2. Bunlar ayrı notebook config koşularıdır.
4. Eğitimde `--train-only` veya notebook'ta `PROTOCOL_FROZEN=False`. Checkpoint seçimi yalnız validation ile yapılır. Baseline temiz doğruluğu, adversarial modeller PGD doğruluğunu en yüksek yapan checkpoint'i seçer. Tam eşitlikte önceki checkpoint korunur.
5. Ayarlar dondurulduktan sonra bütün checkpoint'lerde açıkça `--eval-steps 100 --eval-restarts 5` kullan. Notebook'ta `PROTOCOL_FROZEN=True` ve `FINAL_STRONG_EVAL=True` seç. Tarihî config'lerin varsayılan10 restart değerini final protokol sanma.
6. Üç seed'in bütün değerlerini, ortalama ve **örnek std (ddof=1)** ile raporla. Farklı protokolleri veya kaynak sürümlerini tek ortalamada toplama.

Free GPU oturumlarında Drive kaydı ve `EPOCHS_PER_CALL=1` kullanılabilir. Son tamamlanan epoch'tan devam edilir; kesilen epoch baştan yürür. Her epoch'un tam state'i `last.pt`, seçilen state `best.pt` olarak tutulur. Yeni değerlendirmeler önceki dosyaları değiştirmeden ayrı klasöre yazılır; `latest_evaluation.json` son konumu gösterir.

## Sonuçların yorumu

CSV'ye göre curriculum PGD100×5 doğruluğu **88.13±0.87%**; baseline **0.00±0.00%**; PGD40 **31.28±34.53%**. Son sayının yüksek yayılımı başarısız seed'leri açık eder. Normal doğruluk ve eğitim kararlılığı birlikte değerlendirilmelidir. Üç seed, yöntemin bütün ağlarda daha üstün olduğunu kanıtlamaz.

512 örnek validation skorlarını 10.000 test skoru ile eşdeğer protokol gibi kıyaslama. AutoAttack'in Square uyarısı ve ek bütçeyle bulunan hatalar saklanmamalı; gradient masking kesin olarak dışlanmış değildir. Matematiksel sertifika iddiası kurulmaz.

Tam eğitim geçmişleri mevcut değilse öğrenme eğrisi uydurma. Ayrı seed'lerin sonuçlarını epoch eğrisi gibi çizme. CSV'deki oranlar `[0,1]` aralığındadır; yüzde grafiğinde 100 ile çarpılır. Numbers'ta ondalık ayracı yanlış görünürse ham CSV'yi değiştirme.

## Deney kaydı

Yeni koşularda kaynak commit/fingerprint, config, seed, cihaz/ortam, checkpoint SHA-256, seçilen epoch, saldırı bütçesi, örnek sayısı ve çıktı konumu birlikte saklanmalı. Ekran görüntüleri anlatımı destekler; sayısal kayıtların yerine geçmez. Eski ölçüm dosyalarının üzerine yeni sonuç yazma.
