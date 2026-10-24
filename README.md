# CMP4509 — MNIST Üzerinde PGD ile Adversarially Robust Learning

[Public GitHub deposu](https://github.com/emirbakac/cmp4509-madry-mnist) · [Colab notebook](https://colab.research.google.com/github/emirbakac/cmp4509-madry-mnist/blob/main/notebooks/01_mnist_reproduction.ipynb) · [Final test sonuçları (CSV)](results/FINAL_TEST_ALL_MODELS.csv) · [Teori notları](docs/THEORY.md)

**CMP4509 Introduction to Deep Learning — Fall 2026, Term Project**  
**Reproduced paper:** Madry et al., *[Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)* (ICLR 2018).

Bu proje, Madry ve arkadaşlarının **PGD adversarial training'in standart eğitime kıyasla adversarial saldırılara karşı dayanıklılığı artırabileceği** yönündeki temel iddiasını MNIST üzerinde, ücretsiz Google Colab GPU bütçesine uygun daha küçük bir PyTorch CNN ile inceler. Üç ayrı eğitim stratejisini, **3 random seed** ve aynı final test protokolü altında karşılaştırır:

1. **Baseline:** Yalnız temiz MNIST görüntüleriyle eğitim.
2. **PGD40 from scratch:** İlk epoch'tan itibaren her eğitim batch'ine 40 adımlı PGD uygulanması.
3. **PGD curriculum:** İlk **3 epoch PGD10**, sonraki **7 epoch PGD40** ile adversarial training.

**Sonuç:** Standart eğitimde güçlü PGD saldırısı altında doğruluk **%0.00 ± 0.00** iken curriculum eğitiminde **%88.13 ± 0.87** oldu (10.000 MNIST test görüntüsü, PGD100 × 5 restart, 3 seed, örnek standart sapma). PGD40 from scratch koşularında üç seed'in ikisinde *single-class prediction collapse* gözlendi. Bu bulgular **bizim ölçeklendirilmiş deney koşullarımız** içindir; orijinal makaledeki mimari, bütçe veya sayılarla birebir eşdeğerlik iddiası değildir.

## 1. Final deney sonuçları

**Test protokolü (model ve saldırı ayarları dondurulduktan sonra):** MNIST'in ayrı **10.000** görüntülük test seti; `L∞`, `ε = 0.3`, saldırı adım boyu `α = 0.01`, **PGD100 × 5 random restarts**. Eğitim sırasında test seti model veya hiperparametre seçmek için kullanılmadı. Her koşuda validation ile seçilen `best.pt` değerlendirildi. Aşağıdaki `±` işareti, 3 bağımsız training seed'i (0, 1, 2) üzerinden **sample standard deviation (ddof=1)** anlamına gelir; güven aralığı değildir.

| Eğitim yöntemi | Clean test accuracy | FGSM test accuracy | PGD100 × 5 test accuracy |
|---|---:|---:|---:|
| Baseline | **99.03% ± 0.05** | 11.49% ± 1.25 | **0.00% ± 0.00** |
| PGD40 from scratch | 38.12% ± 46.37 | 34.41% ± 39.95 | 31.28% ± 34.53 |
| **PGD10 → PGD40 curriculum** | **97.85% ± 0.30** | **93.91% ± 0.62** | **88.13% ± 0.87** |

### Seed bazında gerçek final test ölçümleri

Yüzde biçiminde gösterilmiştir. Ham CSV, oranları `[0,1]` aralığında ve ilgili checkpoint SHA-256 değerleriyle saklar.

| Yöntem | Seed | Seçilen epoch | Clean | FGSM | PGD100 × 5 |
|---|---:|---:|---:|---:|---:|
| Baseline | 0 | 10 | 99.07% | 12.38% | 0.00% |
| Baseline | 1 | 8 | 99.05% | 12.03% | 0.00% |
| Baseline | 2 | 10 | 98.97% | 10.07% | 0.00% |
| PGD40 | 0 | 10 | 91.66% | 80.54% | 71.15% |
| PGD40 | 1 | 1 | 11.35% | 11.35% | 11.35% |
| PGD40 | 2 | 1 | 11.35% | 11.35% | 11.35% |
| Curriculum | 0 | 10 | 97.50% | 93.22% | 87.62% |
| Curriculum | 1 | 10 | 98.01% | 94.42% | 89.14% |
| Curriculum | 2 | 10 | 98.03% | 94.09% | 87.63% |

**Kaynak veri:** [`results/FINAL_TEST_ALL_MODELS.csv`](results/FINAL_TEST_ALL_MODELS.csv). `checkpoint_sha256` sütunu her ölçümün hangi model dosyasından geldiğini doğrulamak içindir. CSV'yi Excel/Numbers ile açarken yerel ondalık ayracı nedeniyle `0.975` gibi değerlerin `975` görünmesine dikkat edin; ham CSV veya `pandas.read_csv` kullanın.

**Sonuçların yorumu:** Baseline temiz doğrulukta en yüksek değere ulaşsa da bu saldırı altında kırılgandır. PGD40 eğitimi seed 0'da anlamlı dayanıklılık öğrendi; seed 1 ve 2 ise on epoch boyunca tek sınıf tahminine yakın takıldı. Bu iki seed'in PGD accuracy'sinin %11.35 olması *gerçek dayanıklılık* başarısı anlamına gelmez; clean accuracy de aynı düzeydedir. Hiçbir başarısız seed ortalamadan çıkarılmadı. Curriculum'un üç seed'de de öğrenmesi bu ayarlar altında optimizasyon kararlılığını destekler; bunun bütün ağlar ve ayarlar için genel bir üstünlük kanıtı olduğu iddia edilmez.

### Ek validation saldırı kontrolleri (final test değil)

Curriculum checkpoint'leri seçildikten sonra **aynı 512 validation görüntüsünde**, `ε=0.3` ile ayrıca şu kontroller yapıldı:

| Değerlendirme | Seed 0 | Seed 1 | Seed 2 | Ortalama ± std |
|---|---:|---:|---:|---:|
| PGD100 × 20 restart | 84.38% | 86.52% | 83.98% | 84.96% ± 1.37 |
| Full **standard AutoAttack** (APGD-CE, APGD-T, FAB-T, Square) | 82.62% | 85.16% | 82.23% | 83.34% ± 1.59 |

Seed 0 için artırılmış Square Attack bütçesiyle full AutoAttack sonucu **%82.42** oldu (standart bütçede %82.62). AutoAttack, üç seed'de de Square Attack'in ek örnekler bulduğuna ilişkin uyarı verdi; bu yüzden gradient masking veya daha güçlü/adaptive saldırı olasılığının kesin olarak dışlandığı söylenmez. Bu değerler **512-image validation** kontrolüdür; 10.000 örneklik final PGD test tablosuyla karıştırılmamalıdır. AutoAttack ayrıca kurulan bir dış pakettir; bu depodaki temel test script'inin parçası değildir.

## 2. Yöntem ve tehdit modeli

Temel robust optimizasyon hedefi:

\[
\min_{\theta}\;\mathbb{E}_{(x,y)}\left[
\max_{\delta\in S(x)}\mathcal{L}(f_{\theta}(x+\delta),y)
\right],\quad
S(x)=\{\delta:\|\delta\|_{\infty}\leq 0.3,\ x+\delta\in[0,1]^d\}.
\]

- **İçteki maksimum:** Ağırlıklar sabitken izinli bir adversarial görüntüyle cross-entropy kaybını artırmaya çalışır.
- **Dıştaki minimum:** Saldırılı görüntüler üzerinde model ağırlıklarını Adam ile günceller.
- **FGSM:** İşaret gradyanıyla tek adımlı saldırıdır.
- **PGD:** Rastgele başlangıçtan birden çok işaret-gradyan adımı atar; her adımda `L∞` topuna ve `[0,1]` piksel aralığına projeksiyon uygular.
- **İki ayrı türev:** Saldırı üretiminde `∇x loss`; eğitimde sabit kabul edilen saldırılı girdi üzerinden `∇θ loss` alınır. Saldırı grafiği `detach()` ile ayrılır.
- **Robust accuracy:** Temizken yanlış sınıflanan örnekler dayanıklı sayılmaz. Evaluator; temiz örnek, PGD ara iterasyonları ve bütün restart'lardaki başarılı saldırıları örnek bazında korur. Restart doğrulukları ortalanmaz.

Modelin bir saldırı tarafından kırılamaması **matematiksel sertifikalı dayanıklılık** değildir. Denklem türetimi ve kodun mantığı için [`docs/THEORY.md`](docs/THEORY.md) bölümüne bakın.

## 3. Veri, ağ ve eğitim ayarları

| Bileşen | Bu projede kullanılan ayar |
|---|---|
| Veri | MNIST; **55.000 train / 5.000 validation / 10.000 test** |
| Bölme | `split_seed = 2026`, tüm training seed'lerinde aynı validation split |
| Ön işleme | `ToTensor()`, ham `[0,1]` pikseller; ek normalize yok |
| CNN | `Conv(1→16, 5×5) → ReLU → MaxPool(2) → Conv(16→32, 5×5) → ReLU → MaxPool(2) → Flatten → Linear(1568→128) → ReLU → Linear(128→10)` |
| Parametre | **215.370** öğrenilebilir parametre |
| Optimizer | Adam, learning rate `0.001` (diğer optimizer hiperparametreleri PyTorch varsayılanı) |
| Batch / epoch | `128`, **10 epoch** (`main` ve curriculum) |
| Seeds | `0`, `1`, `2` (training runs); validation attack RNG için `eval_seed=12345` |
| Training PGD | `ε=0.3`, `α=0.01`, 1 random start; PGD40 veya curriculum için PGD10→PGD40 |
| Validation PGD | 40 adım, 1 restart; her epoch'ta aynı validation protokolü |
| Model seçimi | Baseline: en iyi **clean validation**; adversarial modeller: en iyi **PGD validation** accuracy |
| Nihai test | Clean, FGSM ve `PGD100 × 5`; 10.000 örnek, checkpoint `best.pt` |
| Compute | Ücretsiz **Google Colab T4** GPU; oturumlar arasında Google Drive checkpoint'leriyle devam |

### Makaleden bilinçli sapmalar

[MadryLab'ın MNIST uygulaması](https://github.com/MadryLab/mnist_challenge), 32/64 convolution kanal ve 1024 hidden-unit içeren daha büyük bir ağ; yaklaşık `1e-4` öğrenme oranı, batch 50 ve daha uzun eğitim bütçesi kullanır. Bizim **16/32 kanal, 128 hidden**, `1e-3`, batch 128 ve 10 epoch seçimlerimiz ücretsiz GPU üzerinde tamamlanabilir bir deney hedefler. PyTorch ağırlık başlangıcı da resmî TensorFlow uygulamasından farklıdır. Ayrıca **PGD10→PGD40 curriculum**, Madry makalesinin birebir eğitim protokolü değil, ilk PGD40 koşularında görülen collapse'ı incelemek için oluşturulmuş ek kontrollü deneydir. Bu sapmalar nihai sonuçların makalenin sayılarıyla doğrudan eşitlenmesini engeller.

### Curriculum'un deneysel geçmişi

Gerçekte önce `main_pgd10_probe` ile seed 0/1/2 için **3 epoch PGD10** eğitildi; model ağırlıkları, Adam optimizer durumu ve RNG kaydı korunarak ayrı `main_curriculum_pgd10_to_pgd40` koşularına aktarıldı. **Epoch 4–10 PGD40** ile tamamlandı. Başka bir eğitim protokolü uygulanmış gibi sunulmamalıdır. Güncel `experiment.py` aynı **3+7 eğitim adımı programını sıfırdan** uygulayabilmektedir; ayrı config ve klasör kullanır. Donanım/kütüphane sürümlerinden dolayı bit-düzeyinde aynı checkpoint garanti edilmez.

## 4. Repository yapısı

```text
configs/
  smoke.json
  pilot.json
  main.json
  main_pgd10_probe.json
  main_curriculum_pgd10_to_pgd40.json
mnist_robust/
  data.py             # Sabit MNIST split / DataLoader
  model.py            # SmallCNN
  attacks.py          # FGSM ve L-infinity PGD
  experiment.py       # Eğitim, resume, validation ve final evaluation
  summarize.py        # Seed bazında ortalama ± sample SD ve görseller
  utils.py            # RNG, device, ortam kaydı
notebooks/
  01_mnist_reproduction.ipynb
  # Kaynak kodun gömülü olduğu notebook; değiştirilince yeniden oluşturulur
tools/
  build_notebook.py   # Mevcut kaynaklardan notebook üretir
tests/                 # Pipeline ve attack invariant testleri
docs/
  THEORY.md
  EXPERIMENT_PLAN.md
  COURSE_DELIVERABLES.md
  VALIDATION.md
  references.bib
results/
  FINAL_TEST_ALL_MODELS.csv  # 9 model × final PGD100x5 (gerçek test)
  smoke_check/               # Bilimsel sonuç olmayan yazılım kontrolü
runs/                        # Çalıştırınca oluşur (Git'e eklenmez)
```

## 5. Nasıl çalıştırılır?

### A. Colab / Kaggle notebook

1. [Notebook'u Colab'da açın](https://colab.research.google.com/github/emirbakac/cmp4509-madry-mnist/blob/main/notebooks/01_mnist_reproduction.ipynb). **Runtime → Change runtime type → GPU** (Kaggle: Accelerator → GPU) seçin. GPU kullanım kotası değişkendir.
2. İlk çalıştırmada `CONFIG_NAME="smoke"` ile veri→model→saldırı→checkpoint→CSV zincirini doğrulayın. **Smoke sonuçları bilimsel MNIST kanıtı değildir.**
3. Asıl bilimsel koşular için sırasıyla `CONFIG_NAME="main"` (`MODES`: baseline/adversarial), `CONFIG_NAME="main_pgd10_probe"` ve `CONFIG_NAME="main_curriculum_pgd10_to_pgd40"` (`MODES`: adversarial) kullanılır. Her scientific config için **seed 0/1/2** gereklidir; bunlar aynı anda tek bir config'ten üretilmez. Güncel `tools/build_notebook.py` bu seçenekleri destekler.
4. Ücretsiz GPU oturumlarında `USE_GOOGLE_DRIVE=True`, `RESUME=True` ve gerekirse `EPOCHS_PER_CALL=1` kullanın. Eğitimde `PROTOCOL_FROZEN=False` ve test kapalı tutulur. Model seçimi validation üzerinden yapılır.
5. **Test protokolü dondurulduktan sonra:** `PROTOCOL_FROZEN=True` ve `FINAL_STRONG_EVAL=True` ile PGD100 × 5 uygularsınız. Raporlanan test sonuçları tek, sabit protokole aittir. Model veya saldırı ayarını test skoruna bakarak değiştirmeyin.

**Önemli:** `notebooks/01_mnist_reproduction.ipynb`, `tools/build_notebook.py` tarafından kaynakları içine yazılarak üretilir. Kaynak/config/README değiştiyse **notebook'u da yeniden oluşturup commit edin**; aksi hâlde notebook eski kodu yazabilir. Sonuç CSV'sinin mevcut olması, notebook'un güncel olduğu anlamına gelmez.

### B. Komut satırı ile yeniden üretim

Python/PyTorch kurulumu (özellikle Colab'da sağlanan CUDA ile uyumlu `torch`/`torchvision` çiftini koruyun):

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

**Önce eğitim (test setini model seçimine açmadan):**

```bash
# Standard baseline, seed 0
python -m mnist_robust.experiment --config configs/main.json --mode baseline --seed 0 --output-dir runs --device auto --train-only

# PGD40 from scratch, seed 0
python -m mnist_robust.experiment --config configs/main.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only

# PGD10 probe, seed 0 (3 epochs)
python -m mnist_robust.experiment --config configs/main_pgd10_probe.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only

# Two-stage PGD10 (epochs 1-3) + PGD40 (epochs 4-10), seed 0
python -m mnist_robust.experiment --config configs/main_curriculum_pgd10_to_pgd40.json --mode adversarial --seed 0 --output-dir runs --device auto --train-only
```

Bu dört örnekte `--seed 0` yerine **1 ve 2** de kullanın. İlk iki eğitim modu ve curriculum final karşılaştırmadadır; `main_pgd10_probe` yalnızca curriculum başlangıcını inceleyen ek deneydir. Modeli ayrı koşu klasörlerinde eğitin. Kesintide aynı ayarla `--resume` ekleyin; epoch başına checkpoint almak için `--epochs-per-call 1`, kalan bütün epoch'ları yürütmek için `--epochs-per-call 0` kullanabilirsiniz.

**Dondurulmuş test örneği (curriculum seed 0):**

```bash
python -m mnist_robust.experiment \
  --config configs/main_curriculum_pgd10_to_pgd40.json \
  --mode adversarial --seed 0 --output-dir runs \
  --device auto --eval-only \
  --checkpoint runs/main_curriculum_pgd10_to_pgd40/adversarial/seed_0/best.pt \
  --eval-steps 100 --eval-restarts 5
```

Baseline ve PGD40 checkpoint'leri için config/mode/seed/path'i uygun şekilde değiştirin. Test, önceden sabitlenmiş protokol ile **yalnızca nihai ölçüm** için kullanılmalıdır; yeniden test edip sonuçlara göre hiperparametre ayarı yapılmaz.

**Özet ve grafikler:**

```bash
python -m mnist_robust.summarize --runs runs/main --output results/main
python -m mnist_robust.summarize --runs runs/main_curriculum_pgd10_to_pgd40 --output results/curriculum
# Kaynakları / README / config'leri düzenlediyseniz güncel notebook üretin:
python tools/build_notebook.py
```

Özetleyici aynı saldırı/config/protokol sonuçlarını birleştirir; farklı PGD restart bütçelerini veya smoke koşularını aynı istatistiğe katmaz. Kaynak kod, config ve saldırı protokolünün bütünlüğü kontrol edilmelidir. CSV değerlerini yüzdeye çevirmek için 100 ile çarpın.

## 6. Kaydedilen çıktılar ve tekrarlanabilirlik

Her koşuda `runs/{config}/{mode}/seed_{seed}/` klasöründe eğitim geçmişi (`history.csv`), config/ortam bilgisi, `last.pt` (resume) ve validation ile seçilen `best.pt` tutulur. Değerlendirmeler `evaluations/pgd{steps}_r{restarts}/` altındadır; `evaluation.json`, `metrics.csv`, `per_example.csv` ve temiz/FGSM/PGD örnek görseli üretilir. Final rapor tablosunun küçük, paylaşılabilir sürümü **[`results/FINAL_TEST_ALL_MODELS.csv`](results/FINAL_TEST_ALL_MODELS.csv)** dosyasıdır. Büyük checkpoint'ler, MNIST ham verisi ve kişisel Drive klasörleri GitHub'a yüklenmez.

`main` checkpoint seçimi baseline'da **clean validation**, PGD40'ta **PGD40 validation** üzerinden yapılır. Curriculum için de **PGD40 validation** kullanılır. İki başarısız PGD40 seed'i dahil olmak üzere önceden belirlenen tüm seed'ler raporlanır. `seed` tek başına farklı PyTorch/CUDA ve GPU sürümlerinde bit-düzeyinde aynı sonucu garanti etmez; gerçek çalışma ortamı kayıt dosyalarında saklanır.

Sonuçlar [yazarların MNIST Challenge kodu](https://github.com/MadryLab/mnist_challenge) ve asıl [makale](https://arxiv.org/abs/1706.06083) ile yöntem bakımından karşılaştırılır; model kapasitesi ve hesaplama bütçesi farklı olduğundan rakamlar doğrudan eşit koşullarda değildir. Eğitim başarısızlığı veya robust accuracy düşüşleri saklanmaz; **raporun Discussion bölümünde** açıklanır.

## 7. Ders teslimleri

- **GitHub:** Public, kendi grubunuza ait depo; README, PyTorch kodu, yeniden üretilebilir notebook, config dosyaları ve gerçek sonuçlar. GitHub adresi raporda bulunmalıdır.
- **Rapor:** Teams'teki **`CMP4509_report_template.zip`** içinde verilen LaTeX şablonuyla yazılacak ve **PDF** olarak **13 Aralık 2026, 23:59**'a kadar teslim edilecektir. `\showguidesfalse` teslim öncesi ayarlanmalı; **Contributions** ve **Use of AI Tools** bölümleri doldurulmalıdır.
- **Sunum:** **30 Kasım veya 7 Aralık 2026**, soru-cevap dahil **10 dakika**, hedef yaklaşık **8–10 slayt**. Üç üyenin tamamı PGD/min–max türetimini ve deneyleri savunabilmelidir.

Rapor, üniversitenin gerçek ders şablonu kullanılarak oluşturulmalıdır; README veya genel amaçlı başka bir LaTeX şablonu onun yerini almaz. Ders yönergesine göre public depo eksikliği, ders şablonu dışında hazırlanmış rapor veya PDF olmayan teslim projeden **0 puan** alınmasına yol açar.

## 8. Kaynaklar

- Madry, A. et al. (2018), [*Towards Deep Learning Models Resistant to Adversarial Attacks*](https://arxiv.org/abs/1706.06083), ICLR.
- [MadryLab MNIST Challenge (official implementation)](https://github.com/MadryLab/mnist_challenge).
- Goodfellow, I. et al. (2015), [*Explaining and Harnessing Adversarial Examples*](https://arxiv.org/abs/1412.6572) — FGSM.
- Croce, F. and Hein, M. (2020), [*Reliable Evaluation of Adversarial Robustness with an Ensemble of Diverse Parameter-free Attacks*](https://arxiv.org/abs/2003.01690) — AutoAttack.
- Projede kullanılan kaynaklar ve BibTeX: [`docs/references.bib`](docs/references.bib).

**Not:** Bu çalışma, robust accuracy'yi verilen PGD/AutoAttack saldırı protokolleri altında ampirik olarak ölçer; bütün olası adversarial perturbation'lara karşı güvenlik garantisi sunmaz.
