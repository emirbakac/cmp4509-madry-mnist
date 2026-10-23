# MNIST üzerinde PGD ile dayanıklı öğrenme

[Public GitHub deposu](https://github.com/emirbakac/cmp4509-madry-mnist) · [Notebook’u Colab’da aç](https://colab.research.google.com/github/emirbakac/cmp4509-madry-mnist/blob/main/notebooks/01_mnist_reproduction.ipynb)

CMP4509 Deep Learning grup projesi: Madry ve arkadaşlarının **“Towards Deep Learning Models Resistant to Adversarial Attacks”** çalışmasının ana fikrini, ücretsiz Colab/Kaggle GPU'larında çalışabilecek küçük bir PyTorch deneyiyle sınamak.

**Araştırma sorumuz:** Aynı ağı normal eğitimle ve PGD adversarial training ile eğittiğimizde, temiz görüntü doğruluğu ve güçlü PGD saldırısı altındaki doğruluk nasıl değişiyor?

Bu depo bir başlangıç uygulamasıdır. Eğitim yapılmadan sonuç uydurulmaz. Küçük ağ, eğitim bütçesi ve doğrulama bölümü nedeniyle **makaledeki sayıları birebir tekrar ettiğimizi iddia etmiyoruz**. Amaç, ana sonucu kontrollü ve tekrar edilebilir biçimde incelemek.

## İlk oturum: küçük bir deneyi uçtan uca bitirelim

1. `notebooks/01_mnist_reproduction.ipynb` dosyasını Colab'a yükle veya Kaggle Notebook olarak içe aktar. Notebook bütün kaynak dosyalarını görünür hücrelerde içerir; başka bir depoyu klonlamaya gerek yoktur.
2. Colab'da **Runtime → Change runtime type → GPU**, Kaggle'da **Settings → Accelerator → GPU** seç. Kaggle'da MNIST indirmek için Internet açık olmalı. GPU kotası ve erişim değişebilir.
3. Varsayılan `CONFIG_NAME = "smoke"` ayarını koru ve hücreleri yukarıdan aşağı çalıştır. Bu küçük koşu önce normal eğitimi, sonra PGD eğitimini ve değerlendirmeyi sınar. CPU'da da çalışabilir.
4. İlk sonuç tablosunu ve temiz/saldırılmış örnekleri kontrol et. **Ayarlar, kullanılan cihaz ve sonuç tablosunun ekran görüntüsünü al.** Bunlar ileride sunumun deney sürecini anlatacak.
5. Notebook'un sonundaki arşivi indir. Colab oturumu kapanınca geçici dosyalar silinebilir; uzun koşudan önce Drive kaydını aç. Kaggle'da **Save Version** ile çıktılarını sakla.

Smoke test bilimsel sonuç değildir: 256 eğitim örneği ve tek epoch yalnızca veri, gradyan, saldırı, kayıt ve raporlama zincirini kontrol eder.

Min–max, FGSM türetimi, PGD güncellemesi ve iki farklı gradyanı adım adım anlamak için [kısa teori rehberini](docs/THEORY.md) oku. Rapor için [BibTeX kaynakları](docs/references.bib) da hazır.

## Fikir: saldırganın bulduğu zor örneklerle öğrenmek

Makalenin hedefi şudur:

\[
\min_{\theta}\;\mathbb{E}_{(x,y)}\left[\max_{\|\delta\|_{\infty}\leq\epsilon}\mathcal{L}(f_\theta(x+\delta),y)\right].
\]

İçteki **max**, mevcut modeli zorlayacak izinli bir piksel değişimini arar. Dıştaki **min**, model ağırlıklarını bu zor örneklerdeki kaybı azaltacak yönde günceller. PGD, içteki aramaya pratik bir yaklaşım sağlar; gerçek en kötü saldırıyı bulduğunu garanti etmez.

- MNIST piksel aralığı `[0, 1]`; tehdit sınırı **L∞, ε = 0.3**. Görüntülere ayrıca ortalama/standart sapma normalizasyonu uygulanmaz, böylece epsilon doğrudan bu piksel ölçeğindedir.
- **FGSM:** kaybı büyüten yönde tek adım.
- **PGD:** küçük adımlar, her adımda orijinal görüntünün izinli ε komşuluğuna ve `[0, 1]` aralığına geri yansıtma.
- **Rastgele başlangıç:** her restart saldırıyı izinli bölgenin başka bir noktasından başlatır.
- **Güçlü değerlendirme:** eğitimde kullanılandan daha fazla PGD adımı ve birden fazla restart. Bir örneği herhangi bir restart/adım bozabiliyorsa değerlendirmede bu başarısızlık korunur; restart doğruluklarının ortalaması alınmaz.

Ölçtüğümüz dayanıklılık, seçilen saldırı ve bütçe altındaki ampirik doğruluktur; matematiksel bir dayanıklılık sertifikası değildir.

## Resmî uygulamadan aldıklarımız ve değiştirdiklerimiz

Kaynaklar: [makale](https://arxiv.org/abs/1706.06083), [yazarların MNIST Challenge uygulaması](https://github.com/MadryLab/mnist_challenge), [resmî ayarlar](https://github.com/MadryLab/mnist_challenge/blob/master/config.json), [resmî model](https://github.com/MadryLab/mnist_challenge/blob/master/model.py), [resmî PGD saldırısı](https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py).

| Seçim | Yazarların MNIST uygulaması | Bu projenin `main` başlangıç ayarı |
|---|---|---|
| Piksel ölçeği / ε | `[0,1]` / `0.3` | Aynı |
| Eğitim saldırısı | PGD, 40 adım, adım boyu `0.01`, rastgele başlangıç, cross-entropy | Aynı saldırı ailesi ve değerler |
| Ağ | 5×5 convolution: 32 ve 64 kanal; max-pooling; 1024 birimli gizli katman | Aynı 2-convolution düzeni; 16 ve 32 kanal; 128 birimli gizli katman |
| Optimizasyon | Adam, `1e-4`, batch 50, 100.000 güncelleme | Adam, `1e-3`, batch 128, 10 epoch |
| Veri kullanımı | Resmî kodun eğitim akışı | 55.000 eğitim / sabit 5.000 doğrulama / 10.000 test |
| Başlatma | Resmî TensorFlow ağırlık/bias başlatması | PyTorch varsayılan başlatması |
| Değerlendirme | Makalede ve challenge'da farklı saldırı bütçeleri raporlanır | `main`: PGD 100 × 10 restart; dondurulmuş final: 100 × 20 restart |

Resmî konfigürasyondaki PGD 40 ayarı tek başına yeterli bir final değerlendirme standardı sayılmaz. Makalenin güçlü saldırı değerlendirmesini örnek alarak eğitimden güçlü bir test kullanıyoruz. Daha büyük ağ/uzun eğitim sonradan deney olabilir; önce küçük koşuyu gerçekten tamamlayalım.

## Dosyalar

```text
configs/                  smoke, pilot ve main JSON ayarları
mnist_robust/              veri, model, saldırılar, eğitim, değerlendirme ve özetleme
notebooks/
  01_mnist_reproduction.ipynb  kendi kaynaklarını içeren, baştan sona notebook
tests/                    saldırı sınırları ve ilgili doğrulamalar
docs/
  THEORY.md                min–max, FGSM/PGD ve metriklerin açıklaması
  references.bib           birincil kaynaklar
  EXPERIMENT_PLAN.md       deney sırası, ölçümler ve yorumlama
  COURSE_DELIVERABLES.md   ders teslimleri ve ekran görüntüsü kontrol listesi
runs/                     çalışınca oluşur: checkpoint, geçmiş, ayarlar, metrikler
results/                  özet CSV, tablolar ve şekiller
```

## Yerel kullanım veya terminalden çalışma

Yeni bir Python ortamında, projenin kök klasöründen:

```bash
python -m pip install -r requirements.txt
python -m mnist_robust.experiment --config configs/smoke.json --mode baseline --seed 0 --output-dir runs --device auto
python -m mnist_robust.experiment --config configs/smoke.json --mode adversarial --seed 0 --output-dir runs --device auto
python -m mnist_robust.summarize --runs runs --output results --allow-single-seed
```

Colab/Kaggle'ın önceden kurulu GPU destekli PyTorch paketini gereksiz yere değiştirme. Notebook yalnızca gerekli eksik paketleri kurar; uyumsuz bir kurulumda açık hata verir. Aynı sonuçları karşılaştırırken kaydedilmiş Python/PyTorch/CUDA ortam bilgisini de sakla.

İnternet veya MNIST erişimi yoksa yalnızca kod yolunu kontrol etmek için `--synthetic` eklenebilir. Sentetik veri koşuları MNIST sonucu değildir ve gerçek veriyle birleştirilmez.

## Deneyi büyütme sırası

| Ayar | Eğitim / doğrulama / test örneği | Epoch | Eğitim PGD | Test PGD | Kullanım |
|---|---:|---:|---:|---:|---|
| `smoke` | 256 / 64 / 64 | 1 | 2 | 5 × 2 restart | Kodun çalıştığını doğrula |
| `pilot` | 10.000 / 1.000 / 1.000 | 3 | 10 | 40 × 3 restart | GPU maliyetini ve öğrenmeyi gözle |
| `main` | 55.000 / 5.000 / 10.000 | 10 | 40 | 100 × 10 restart | Üç seed ile asıl karşılaştırma |
| Dondurulmuş final | `main` test kümesi | Yeni eğitim yok | — | 100 × 20 restart | Seçilen checkpoint'leri güçlü test et |

Bunlar ücretsiz GPU için **ölçeklenebilir başlangıç bütçeleridir**; verilen sürede tamamlanacakları garanti değildir. `main` PGD eğitimi ve özellikle 20 restart test pahalıdır. Önce pilot süresini ölç, sonra süre/örnek sayısını kaydet. Bütçeyi değiştirirsen yeni ayarı ayrı isimle sakla ve hangi protokolün kullanıldığını rapora yaz.

Model ve eğitim ayarlarını **doğrulama kümesiyle** seç. Test setini görmek için ayarları dondurmayı bekle. Pilotun test sonuçlarına bakarak ana modeli ayarlama; pilot koşularında `--train-only` kullan:

```bash
python -m mnist_robust.experiment --config configs/pilot.json --mode baseline --seed 0 --output-dir runs --device auto --train-only
python -m mnist_robust.experiment --config configs/pilot.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only
```

Aynı düzeni `main` için **seed 0, 1, 2** ve iki eğitim türünde çalıştır. Notebook bunu döngüyle yapar. Normal eğitimde en iyi temiz doğrulama checkpoint'i, PGD eğitiminde en iyi saldırılı doğrulama checkpoint'i seçilir; karşılaştırmada bu seçim kuralını açıkça belirt.

Kesilen bir eğitim aynı komuta `--resume` eklenerek son tamamlanmış checkpoint'ten sürdürülebilir. Oturumu kısa parçalara bölmek için `--epochs-per-call 1` kullanabilirsin; bütün epoch'lar bitmeden otomatik test başlamaz. Veri, model ve eğitim ayarlarını değiştirdikten sonra eski checkpoint'e devam etme; ayrı koşu başlat.

Ayarları dondurduktan sonra örneğin PGD eğitimli seed 0 modelini güçlü test et:

```bash
python -m mnist_robust.experiment --config configs/main.json --mode adversarial --seed 0 --output-dir runs --device auto --eval-only --checkpoint runs/main/adversarial/seed_0/best.pt --eval-steps 100 --eval-restarts 20
```

Aynı testi `baseline` ve diğer seed'ler için de uygula. Sonrasında:

```bash
python -m mnist_robust.summarize --runs runs/main --output results/main
```

Özetleyici farklı veri/config/saldırı protokollerini ayrı tutar. `smoke`, `pilot`, `main` veya farklı restart sayıları aynı ortalamaya girmemelidir. Tek seed ile örnek çıktı görmek için `--allow-single-seed` vardır; tek seed için standart sapma güvenilir biçimde hesaplanamaz. Ana raporda **her model için üç seed**, seed başına değerler ve **ortalama ± örnek standart sapma** bulunmalı.

## Neler kaydediliyor?

Her koşu `runs/{config}/{mode}/seed_{seed}/` altında ayarları, ortam bilgisini, `last.pt`, `best.pt`, eğitim geçmişini ve test yapıldıysa metrikleri/örnek görselini saklar. Sentetik smoke klasöründe `smoke_synthetic` adı kullanılır. Farklı test bütçeleri `evaluations/pgd{steps}_r{restarts}/` altında ayrı korunur; koşu kökündeki dosyalar son değerlendirmeyi gösterir. `last.pt` devam etmek, `best.pt` değerlendirme için kullanılır. Toplu özetleyici `per_seed.csv`, `summary.csv`, `summary.md`, `accuracy.png`, `training_curves.png` üretir.

Temiz, FGSM ve PGD doğruluklarını birlikte incele. Yalnız temiz doğruluk, saldırıya dayanıklılığı göstermez. PGD adım/restart sayısını artırınca dayanıklılık belirgin düşüyorsa zayıf test fazla iyimser sonuç vermiştir. Sadece başarılı görünen seed'i seçme; bütün önceden belirlenmiş seed'leri raporla.

**Şu an ölçülmüş bilimsel sonuç yok.** Çalıştırılan smoke testler yalnızca çalışma doğrulamasıdır. Kullanıcı GPU eğitimini yaptıktan sonra asıl tablolar, şekiller ve rapor bulguları üretilecek.

## Ders teslimleri

- **Public GitHub deposu:** kaynaklar, bağımsız notebook, README, deney ayarları ve yeniden üretim adımları. Büyük checkpoint/veri dosyalarını Git geçmişine koyma; paylaşılan çıktılar için ayrıca bağlantı kullan.
- **Rapor:** Teams'teki `CMP4509_report_template.zip` kullanılarak hazırlanmış PDF; **13 Aralık 2026, 23:59 (Türkiye saati)**. Rapor içinde public depo bağlantısı bulunmalı.
- **Sunum:** **30 Kasım veya 7 Aralık 2026**, soru-cevap dahil **10 dakika**, hedef **8–10 slayt**. Her grup üyesi bütün projeyi açıklayabilmeli.

Ders şablonu bu çalışma alanında henüz yok; şablon gelmeden ders formatında tamamlanmış rapor varmış gibi davranılmayacak. Ayrıntılar için [deney planı](docs/EXPERIMENT_PLAN.md) ve [teslim kontrol listesi](docs/COURSE_DELIVERABLES.md).

Yerel kontrollerin kapsamı ve henüz çalıştırılmayan deneyler: [doğrulama kaydı](docs/VALIDATION.md).

Kaynak dosyalarını değiştirdikten sonra bağımsız notebook’u eşitle: `python tools/build_notebook.py`. Gerçek, yalnız kontrol amaçlı çıktılar [smoke_check](results/smoke_check/README.md) altında; bilimsel sonuç değildir.
