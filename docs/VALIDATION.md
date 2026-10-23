# Yerel doğrulama kaydı — 24 Ekim 2026

Bu kayıt yazılım kontrolüdür; makalenin ana sonucunun tekrarlandığı anlamına gelmez.

- `python -m unittest discover -s tests -v`: **17 test geçti**.
- Gerçek MNIST ile `smoke`, seed 0: baseline ve PGD adversarial eğitim, temiz/FGSM/PGD değerlendirme, checkpoint, CSV ve görseller başarıyla üretildi (CPU).
- Bağımsız notebook'un bütün çalışma hücreleri yukarıdan aşağı, hata olmadan çalıştırıldı. İzole klasörde kaynaklar notebook tarafından oluşturuldu; daha önce indirilen gerçek MNIST önbelleği kullanıldı. Notebook `auto` cihaz seçimiyle Apple MPS üzerinde çalıştı. İlk sürüm 30 hücreydi; üç kaynak-belge hücresi eklenen son sürüm 33 hücredir.
- Epoch sınırında kesilip devam edilen CPU adversarial eğitim, kesintisiz eğitimle aynı ağırlıkları verdi.
- Saldırı sınırları, temiz girişteki hataların korunması, restart/ara-adım başarısızlıklarının kaybolmaması, model modunun ve gradientlerin korunması test edildi.
- Farklı protokollerin ayrı gruplanması, örnek standart sapması (`ddof=1`) ve yinelenen seed reddi doğrulandı.

Test ortamı: Python 3.13.0, PyTorch 2.14.1, torchvision 0.29.1, NumPy 2.4.4, pandas 3.0.2, matplotlib 3.10.9. İlk 13 çekirdek test ayrıca PyTorch 2.11.0 CPU ile geçti. Bu sürümleri Colab'a zorla kurmayın; platformun uyumlu CUDA paket çiftini koruyun. Her koşunun `environment.json` ve `sessions.jsonl` dosyalarını saklayın.

Henüz yapılmayanlar: Colab/Kaggle CUDA çalıştırması, pilot/main GPU eğitimleri, üç seed'in bilimsel karşılaştırması, final PGD100×20 testi. MPS'te smoke tamamlandı; MPS/CUDA için bit düzeyinde resume eşitliği ayrıca doğrulanmadı. Birden çok data worker yerelde test edilmedi; varsayılan `num_workers=0`.

Smoke yalnızca 256 eğitim / 64 doğrulama / 64 test örneği, bir epoch kullanır. Başarı oranları ana deney sonucu olarak rapora taşınmamalıdır. Tam deney sonuçları elde edilene kadar raporun bulgular bölümü açık kalır.
