# Kodu anlayarak çalıştırmak: min–max, FGSM ve PGD

Elimizde bir rakam görüntüsü `x`, doğru etiket `y` ve ağırlıkları `θ` olan model `fθ` var. Normal eğitim, temiz görüntülerin cross-entropy kaybını azaltır. Adversarial eğitim, modelin küçük ama kasıtlı değişikliklere karşı da doğru kalmasını hedefler.

## 1. Min–max hedefini okuyalım

\[
\min_\theta\; \mathbb{E}_{(x,y)}\left[
\max_{\delta\in S(x)}\mathcal{L}(f_\theta(x+\delta),y)
\right],
\qquad
S(x)=\{\delta:\|\delta\|_\infty\le\epsilon,\;x+\delta\in[0,1]^d\}.
\]

**İçteki max:** model ağırlıkları sabitken görüntüyü değiştirip kaybı büyüt. **Dıştaki min:** bu zor görüntüyü kullanıp ağırlıkları değiştirerek kaybı azalt. Veri beklentisini minibatch'lerle, içteki aramayı PGD ile yaklaşık çözüyoruz. Madry ve arkadaşlarının temel yaklaşımı dayanıklı optimizasyon çerçevesidir. [Makale](https://arxiv.org/abs/1706.06083)

L∞ uzaklığı, pikseller arasındaki en büyük mutlak farktır: `max(abs(x_adv - x))`. Bizde pikseller `[0,1]`, ε ise `0.3`. Bu sınır toplam değişimi değil **her pikselin** değişimini sınırlar. Çözünürlük, veri ölçeği veya tehdit normu değişirse aynı epsilon başka anlama gelebilir.

## 2. FGSM neden gradyanın işaretini kullanıyor?

Bir görüntünün yakınında kaybı birinci dereceden Taylor yaklaşımıyla yazalım:

\[
\mathcal{L}(x+\delta)\approx\mathcal{L}(x)+g^\top\delta,
\qquad g=\nabla_x\mathcal{L}(f_\theta(x),y).
\]

L∞ topunda bu doğrusal ifadenin maksimumu:

\[
\max_{\|\delta\|_\infty\le\epsilon}g^\top\delta
=\epsilon\sum_i |g_i|=\epsilon\|g\|_1,
\qquad \delta_i^*=\epsilon\,\operatorname{sign}(g_i).
\]

Her pikseli gradyan pozitifse artırmak, negatifse azaltmak doğrusal yaklaşımın kaybını büyütür. L∞ sınırının L1 gradyan normuyla bu ilişkisi **dual norm** ilişkisidir. Görüntü aralığını da koruyarak:

\[
x_{\rm FGSM}=\operatorname{clip}_{[0,1]}
\left(x+\epsilon\operatorname{sign}(g)\right).
\]

Taylor yaklaşımı yaklaşık olduğu için bu tek adım gerçek sinir ağında en kötü görüntüyü garanti etmez. FGSM'nin birincil kaynağı [Goodfellow, Shlens ve Szegedy](https://arxiv.org/abs/1412.6572).

## 3. PGD bu fikri tekrar ediyor

Rastgele bir başlangıç oluştur:

\[
x^{(0)}=\operatorname{clip}_{[0,1]}(x+u),\qquad
u_i\sim\mathcal{U}(-\epsilon,\epsilon).
\]

Sonra her adımda gradyanı yeniden hesapla:

\[
x^{(t+1)}=\Pi_{[x-\epsilon,x+\epsilon]\cap[0,1]^d}
\left(x^{(t)}+\alpha\operatorname{sign}
\big(\nabla_{x^{(t)}}\mathcal{L}(f_\theta(x^{(t)}),y)\big)\right).
\]

`Π` burada piksel bazında kırpmadır. Merkez daima **orijinal x** kalır; merkezi her adım kaydırırsak ε sınırını aşabiliriz. Adım boyu `α`, toplam izin verilen değişim `ε` ile aynı şey değildir. Bu güncelleme ve rastgele başlangıç [yazarların PGD kodunda](https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py) doğrudan görülebilir.

Bir **restart**, yeni bir rastgele başlangıçla saldırıyı baştan çalıştırır. Farklı başlangıçlar farklı zor bölgeler bulabilir. Projemizde değerlendirme bütün başlangıç/ara adım/restart adaylarında bulunan yanlış sınıflandırmayı korur. Son adımın tesadüfen tekrar doğru olması önceki saldırı başarısını silmez. Bu, resmî kodun son adayı döndürme davranışına göre ek bir değerlendirme önlemidir.

## 4. Neden saldırıda `detach`, sonra ayrı eğitim gradyanı?

İki farklı türev alıyoruz:

1. Saldırıda `∇x loss`: model sabit, değişken görüntü. Yeni saldırı adımı `detach()` edilir ve bir sonraki adım için yeniden `requires_grad` açılır. Böylece bütün PGD iterasyonlarını tek dev hesaplama grafiğine bağlamayız.
2. Eğitimde `∇θ loss`: üretilen saldırılı görüntü sabit girdi kabul edilir; yeni ileri geçişte model ağırlıklarına türev alınır ve optimizer günceller.

Bu uygulamada dış adım, PGD algoritmasının bütün hesaplamalarından geriye türev almaz. `detach`, dış kaybın model ağırlıklarına türevini engellemez; yalnız saldırı üretim zincirinden ayırır. Resmî eğitim de önce adversarial örneği üretir, sonra bu örneği eğitim adımına besler. [Resmî eğitim kodu](https://github.com/MadryLab/mnist_challenge/blob/master/train.py)

İç maksimum tam çözülebilseydi, uygun düzenlilik koşullarında bu tür dış türev için Danskin yaklaşımı kullanılabilir. Burada PGD yaklaşık çözücüdür; bu teorik bağlantı kodun küresel optimum bulduğuna dair garanti değildir.

## 5. Doğrulukta payda bütün test seti

`N` test örneği için temiz doğruluk, doğru tahminlerin sayısının `N`'ye bölümüdür. PGD doğruluğunda da payda **aynı N** kalır:

\[
\widehat{A}_{\rm PGD}=\frac1N\sum_{i=1}^N
\mathbf{1}\{f(x_i)=y_i\}
\mathbf{1}\{\text{tutulan bütün PGD adayları doğru sınıflandırıldı}\}.
\]

Yalnız temizken doğru örnekleri paydaya alıp bunu genel dayanıklı doğruluk diye sunmuyoruz. Sıfır değişim de tehdit kümesinde olduğundan temizken yanlış bir örnek dayanıklı olamaz. `combined_accuracy`, temiz + FGSM + PGD adaylarının hepsini birlikte geçme oranıdır. Restart doğruluklarının ortalaması değildir.

Bu metrik **ampirik saldırı doğruluğudur**. PGD'nin hata bulması bir zayıflık gösterir; hata bulamaması izinli bütün değişimlerin güvenli olduğunu kanıtlamaz. Daha fazla adım/restart bu aramayı güçlendirir, sertifikaya dönüştürmez. Sonuçları yorumlarken `ε`, `α`, adım, restart, test örnek sayısı ve seed'i birlikte belirt.

Koddaki karşılıklar: `mnist_robust/attacks.py`, `mnist_robust/experiment.py`. İlk smoke çalışmasından sonra bu dosyaları yukarıdaki iki gradyan ayrımını takip ederek okumak iyi bir ikinci adımdır.
