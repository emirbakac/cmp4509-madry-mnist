"""Build the self-contained teaching notebook from the current project files."""
from pathlib import Path
import json
import textwrap

PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT / "notebooks" / "01_mnist_reproduction.ipynb"
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": textwrap.dedent(text).strip() + "\n"})


md(r"""
# CMP4509 — MNIST üzerinde PGD ile dayanıklı öğrenme

Broski, ilk hedefimiz bütün zinciri küçük bir deneyle çalıştırmak: **veri → CNN → normal eğitim → saldırı → PGD eğitimi → daha güçlü test → CSV/grafik**.

Bu notebook **kendi kaynak kodunu içerir**. Colab/Kaggle'da başka bir depo klonlamadan, yukarıdan aşağı çalışır. Varsayılan `smoke`, küçük bir çalışma kontrolüdür; bilimsel sonuç değildir. Asıl deneyler GPU'da çalıştırılınca gerçek sayıları raporlayacağız.

Kaynaklar: [Madry ve ark., makale](https://arxiv.org/abs/1706.06083), [yazarların MNIST uygulaması](https://github.com/MadryLab/mnist_challenge).

**Colab:** Runtime → Change runtime type → GPU. **Kaggle:** Settings → Accelerator → GPU, ilk MNIST indirmesi için Internet açık. GPU kotası ve erişim garanti değildir. `smoke` CPU'da çalışabilir; `pilot/main` için bu notebook CUDA GPU ister.

Rapor: **13 Aralık 2026, 23:59**; gerçek `CMP4509_report_template.zip` şablonu kullanılacak. Sunum: **30 Kasım veya 7 Aralık**, QA dahil **10 dakika**. Public GitHub bağlantısı raporda bulunmalı.
""")
md("""
## 1 · Ortamı kontrol et

Önceden kurulmuş GPU destekli PyTorch'u koruyoruz. Yalnız eksik paketler kurulur. Birbiriyle uyumsuz torch/torchvision varsa hücre açık hata verir; temiz bir GPU oturumunda yeniden başlatmak genellikle en kolay başlangıçtır.
""")
code(r"""
import importlib.util
import subprocess
import sys

missing_torch = importlib.util.find_spec("torch") is None
missing_vision = importlib.util.find_spec("torchvision") is None
if missing_torch:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.5,<3", "torchvision>=0.20,<1"])
elif missing_vision:
    raise RuntimeError(
        "PyTorch var ama torchvision eksik. Torch sürümünüzle uyumlu torchvision kurun "
        "veya temiz Colab/Kaggle GPU oturumu açın; GPU paketini otomatik değiştirmiyoruz."
    )
for package, requirement in {"numpy": "numpy>=1.26,<3", "matplotlib": "matplotlib>=3.8,<4", "pandas": "pandas>=2.2,<4"}.items():
    if importlib.util.find_spec(package) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

try:
    import torch
    import torchvision
    import numpy as np
    import matplotlib
    import pandas as pd
except Exception as exc:
    raise RuntimeError(
        "Paketler uyumsuz veya eksik. Temiz bir Colab/Kaggle GPU oturumunda yeniden deneyin. "
        "Yerelde PyTorch'un resmî kurulumundan birbiriyle uyumlu torch/torchvision seçin."
    ) from exc

print("Python:", sys.version.split()[0])
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
""")
md("""
## 2 · Deney ayarları ve kalıcı kayıt

İlk çalıştırmada hiçbir şeyi değiştirmeden ilerle. Sonra `pilot` ile öğrenme ve süreyi gözle. `main`, her iki eğitim yöntemi için üç seed kullanır. **Eğitim ve model seçimi doğrulama setiyle yapılır.** `EPOCHS_PER_CALL=1` ile ücretsiz GPU oturumlarında eğitimi birer epoch parçalara ayırabilirsin; test bütün epoch'lar tamamlanana kadar bekler. `pilot/main` testini ancak bütün ayarları dondurunca `PROTOCOL_FROZEN = True` ile aç.

Final rapor için `CONFIG_NAME = "main"`, `PROTOCOL_FROZEN = True`, `FINAL_STRONG_EVAL = True` kullan: 100 PGD adımı × 20 restart. Önceden eğitilmiş checkpoint varsa eğitim son tamamlanan epoch'tan devam eder; tamamlanmışsa yalnız değerlendirme ilerler.

Colab'da uzun koşudan önce `USE_GOOGLE_DRIVE = True` yap. Drive bağlamak kullanıcı girişini gerektirebilir. Kaggle çıktıları `/kaggle/working` altında saklanır; oturum sonunda **Save Version** yap. Aynı kayıt kökünde config'ini değiştirme: değiştirilmiş deney için yeni config adı/kayıt kökü kullan.
""")
code(r"""
from pathlib import Path
import os

CONFIG_NAME = "smoke"                    # "smoke", "pilot", "main"
MODES = ["baseline", "adversarial"]
SEEDS = [0, 1, 2] if CONFIG_NAME in {
    "main",
    "main_pgd10_probe",
    "main_curriculum_pgd10_to_pgd40"
} else [0]
PROTOCOL_FROZEN = False                   # Pilot/main testi: kararlar bittikten sonra True
FINAL_STRONG_EVAL = False                 # Final: PGD100 x 20 restart
USE_GOOGLE_DRIVE = False                  # Colab uzun koşularında True önerilir
RESUME = True                            # last.pt varsa oradan devam et
EPOCHS_PER_CALL = 0                       # 0=hepsi; örn. 1 ile oturum başına bir epoch
SYNTHETIC = False                         # Yalnız çevrimdışı smoke; MNIST sonucu değildir

assert CONFIG_NAME in {
    "smoke",
    "pilot",
    "main",
    "main_pgd10_probe",
    "main_curriculum_pgd10_to_pgd40"
}
assert MODES and set(MODES) <= {"baseline", "adversarial"}
assert SEEDS and len(SEEDS) == len(set(SEEDS))
if CONFIG_NAME != "smoke" and not torch.cuda.is_available():
    raise RuntimeError("pilot/main için CUDA GPU açın veya CONFIG_NAME='smoke' ile başlayın.")
if SYNTHETIC and CONFIG_NAME != "smoke":
    raise ValueError("Sentetik veri yalnız smoke kontrolü içindir.")
if FINAL_STRONG_EVAL and not PROTOCOL_FROZEN:
    raise ValueError("Final test için önce ayarları dondurun ve PROTOCOL_FROZEN=True yapın.")

try:
    IS_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IS_COLAB = False
IS_KAGGLE = Path("/kaggle/working").exists()
if "_NOTEBOOK_BASE_DIR" not in globals():
    _NOTEBOOK_BASE_DIR = Path.cwd()
BASE_DIR = Path("/content") if IS_COLAB else (Path("/kaggle/working") if IS_KAGGLE else _NOTEBOOK_BASE_DIR)
PROJECT_DIR = BASE_DIR / "madry_mnist_project"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)
if USE_GOOGLE_DRIVE:
    if not IS_COLAB:
        raise RuntimeError("USE_GOOGLE_DRIVE yalnız Colab içindir.")
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACT_ROOT = Path("/content/drive/MyDrive/CMP4509_madry")
else:
    ARTIFACT_ROOT = BASE_DIR / "madry_mnist_outputs"
RUN_ROOT = ARTIFACT_ROOT / ("synthetic_runs" if SYNTHETIC else "runs")
RESULTS_DIR = ARTIFACT_ROOT / ("synthetic_results" if SYNTHETIC else "results")
RUN_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_DIR)
for name in ["mnist_robust", "configs", "docs"]:
    Path(name).mkdir(exist_ok=True)

RUN_CONFIG_NAME = CONFIG_NAME + ("_synthetic" if SYNTHETIC else "")
RUN_TEST_EVALUATION = CONFIG_NAME == "smoke" or PROTOCOL_FROZEN
print("Kaynak:", PROJECT_DIR)
print("Kalıcı/indirilecek çıktılar:", ARTIFACT_ROOT)
print("Config:", CONFIG_NAME, "| Modeller:", MODES, "| Seed'ler:", SEEDS)
print("Test açık mı?", RUN_TEST_EVALUATION, "| Final güçlü test?", FINAL_STRONG_EVAL)
print("Ekran görüntüsü zamanı: bu ayarlar ve GPU bilgisi deney kaydınız olsun.")
""")
md(r"""
## 3 · Kaynak kodunu oluştur

Aşağıdaki hücreler projedeki kaynakları görünür biçimde yazar. `Run all` için değiştirmeye gerek yok. Öğrenirken özellikle model, saldırı ve eğitim dosyalarını açıp birlikte okuyabiliriz.

Makalenin hedefi:

\[
\min_\theta \; \mathbb{E}_{(x,y)} [ \max_{\|\delta\|_\infty\leq 0.3} \mathcal{L}(f_\theta(x+\delta),y)] .
\]

İçte PGD, mevcut modeli zorlayacak bir görüntü arar. Dışta optimizer, modelin bu zor görüntüdeki kaybını düşürür. Piksel ölçeği `[0,1]` olduğu için ε=0.3 bu ölçekte uygulanır. Her PGD adımından sonra hem ε sınırına hem `[0,1]` aralığına geri dönülür.

Resmî kod: 32/64 convolution, 1024 gizli birim, Adam `1e-4`, batch 50, 100.000 güncelleme; eğitim PGD40, α=.01, ε=.3. Bizim `main` başlangıcımız daha küçük ağ/10 epoch/Adam `1e-3`/batch128 kullanır. **Bu farklar raporlanır; birebir sayısal tekrar iddiası kurulmaz.**
""")
source_files = sorted((PROJECT / "mnist_robust").glob("*.py")) + sorted((PROJECT / "configs").glob("*.json"))
if not source_files or not list((PROJECT / "mnist_robust").glob("experiment.py")):
    raise SystemExit("Package is not ready; generate only after experiment.py and configs exist.")
# Embed documentation too, so the notebook's downloaded source is usable on its own.
for extra in ["requirements.txt", "README.md", "docs/EXPERIMENT_PLAN.md", "docs/COURSE_DELIVERABLES.md", "docs/THEORY.md", "docs/references.bib", "docs/VALIDATION.md"]:
    p = PROJECT / extra
    if p.exists():
        source_files.append(p)
for p in source_files:
    relative = p.relative_to(PROJECT).as_posix()
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": f"%%writefile {relative}\n" + p.read_text(encoding="utf-8")})
md("""
## 4 · Önce eğit ve doğrula

Her mode/seed ayrı klasöre kaydedilir. Bu hücre **test setini açmaz**. Baseline temiz doğrulama, PGD eğitimli model saldırılı doğrulama üzerinden en iyi checkpoint'i seçer. `last.pt` devam etmek, `best.pt` daha sonra test etmek içindir.

PGD eğitimi aynı epoch sayısında normal eğitimden çok daha pahalıdır. Pilot süresini ölçmeden ana deneyin süresini varsayma. Önbelleğe alınmamış MNIST ilk seferde indirilecektir; sentetik smoke kullanılırsa bunun gerçek veri sonucu olmadığını çıktıda koruyoruz.
""")
code(r"""
import time

def run_command(args):
    print("\nÇalıştırılıyor:", " ".join(str(part) for part in args), flush=True)
    subprocess.run([str(part) for part in args], cwd=PROJECT_DIR, check=True)

training_wall_seconds = {}
for mode in MODES:
    for seed in SEEDS:
        run_dir = RUN_ROOT / RUN_CONFIG_NAME / mode / f"seed_{seed}"
        args = [sys.executable, "-m", "mnist_robust.experiment",
                "--config", f"configs/{CONFIG_NAME}.json", "--mode", mode,
                "--seed", str(seed), "--output-dir", str(RUN_ROOT),
                "--device", "auto", "--train-only",
                "--epochs-per-call", str(EPOCHS_PER_CALL)]
        if RESUME and (run_dir / "last.pt").exists():
            args.append("--resume")
        if SYNTHETIC:
            args.append("--synthetic")
        started = time.perf_counter()
        run_command(args)
        training_wall_seconds[f"{mode}/seed_{seed}"] = time.perf_counter() - started
import csv
config_epochs = __import__("json").loads(Path(f"configs/{CONFIG_NAME}.json").read_text())["epochs"]
TRAINING_COMPLETE = True
for mode in MODES:
    for seed in SEEDS:
        history_path = RUN_ROOT / RUN_CONFIG_NAME / mode / f"seed_{seed}" / "history.csv"
        completed_epochs = len(list(csv.DictReader(history_path.open()))) if history_path.exists() else 0
        TRAINING_COMPLETE &= completed_epochs >= config_epochs
        print(f"{mode}/seed_{seed}: {completed_epochs}/{config_epochs} epoch")
print("Bütün eğitimler tamam mı?", TRAINING_COMPLETE)
print("Bu notebook oturumundaki eğitim/yeniden başlatma süreleri (s):", training_wall_seconds)
print("Ekran görüntüsü: eğitim ve doğrulama çıktıları. Checkpoint'ler:", RUN_ROOT)
""")
md("""
## 5 · Temiz, FGSM ve güçlü PGD testi

Smoke'ta küçük test otomatik açık. Pilot/main'de `PROTOCOL_FROZEN=True` gereklidir; önceki hücrelerin tamamı çalışabilir ama bu izin verilmeden test yapılmaz.

Bir örnek herhangi bir restart/PGD adımında yanlış sınıflandırıldıysa saldırı başarılı sayılır. Restart doğruluklarının ortalamasını almak fazla iyimser sonuç verir. Temizken yanlış olan bir örnek de dayanıklı sayılmaz. FGSM tek adımdır; PGD yinelemelidir. Final PGD100×20, `main` eğitim PGD40'tan güçlüdür ve pahalı olabilir.

Bu test seçilen saldırı altında ampirik dayanıklılık ölçer; bütün olası saldırılara karşı garanti vermez.
""")
code(r"""
EVALUATION_COMPLETED = False
if not TRAINING_COMPLETE:
    print("Eğitim parçası tamamlandı. RESUME=True ile eğitim hücresini tekrar çalıştırın; test bekliyor.")
elif not RUN_TEST_EVALUATION:
    print("Test kapalı: doğrulama sonuçlarıyla karar verin. Sonra PROTOCOL_FROZEN=True yapın.")
else:
    for mode in MODES:
        for seed in SEEDS:
            checkpoint = RUN_ROOT / RUN_CONFIG_NAME / mode / f"seed_{seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Önce eğitim hücresini tamamlayın: {checkpoint}")
            args = [sys.executable, "-m", "mnist_robust.experiment",
                    "--config", f"configs/{CONFIG_NAME}.json", "--mode", mode,
                    "--seed", str(seed), "--output-dir", str(RUN_ROOT), "--device", "auto",
                    "--eval-only", "--checkpoint", str(checkpoint)]
            if SYNTHETIC:
                args.append("--synthetic")
            if FINAL_STRONG_EVAL:
                args.extend(["--eval-steps", "100", "--eval-restarts", "5"])
            run_command(args)
    EVALUATION_COMPLETED = True
    print("Test tamamlandı. Tabloyu görmeden önce config/seed/PGD bütçesini tekrar kontrol edelim.")
""")
md("""
## 6 · CSV, tablolar ve şekiller

Her seed'in değerleri ayrıca saklanır. Ana deneyde **ortalama ± örnek standart sapma** kullanılır. Tek seed'in standart sapması hesaplanabilir bir belirsizlik ölçüsü değildir. Smoke/pilot/main veya farklı saldırı protokolleri aynı ortalamaya girmez.

**Screenshot hatırlatması:** koşul adının göründüğü sonuç tablosu, doğruluk karşılaştırması ve saldırılmış örnekler. CSV'leri de sakla; görüntü dosyası veri dosyasının yerine geçmez. Henüz koşulmamış deneyin sayısını rapora yazmayacağız.
""")
code(r"""
from IPython.display import display, Image, Markdown

if EVALUATION_COMPLETED:
    args = [sys.executable, "-m", "mnist_robust.summarize",
            "--runs", str(RUN_ROOT / RUN_CONFIG_NAME), "--output", str(RESULTS_DIR)]
    if len(SEEDS) < 2:
        args.append("--allow-single-seed")
    run_command(args)
    report = RESULTS_DIR / "summary.md"
    if report.exists():
        display(Markdown(report.read_text(encoding="utf-8")))
    for filename in ["accuracy.png", "training_curves.png"]:
        figure = RESULTS_DIR / filename
        if figure.exists():
            display(Image(filename=str(figure)))
    for mode in MODES:
        example = RUN_ROOT / RUN_CONFIG_NAME / mode / f"seed_{SEEDS[0]}" / "examples.png"
        if example.exists():
            display(Markdown(f"**{CONFIG_NAME} · {mode} · seed {SEEDS[0]}**"))
            display(Image(filename=str(example)))
    print("Sonuç dosyaları:", [p.name for p in sorted(RESULTS_DIR.glob("*"))])
else:
    print("Test tablosu yok. Eğitim geçmişi ve checkpoint'ler saklandı:", RUN_ROOT)
    for history in sorted((RUN_ROOT / RUN_CONFIG_NAME).glob("*/seed_*/history.csv")):
        print("\n", history.relative_to(RUN_ROOT))
        print(history.read_text(encoding="utf-8")[-3000:])
""")
md("""
## 7 · Çalışmayı kaybetmeden sakla

Aşağıdaki arşiv kaynak kodunu, config'leri, checkpoint'leri ve sonuçları içerir. Uzun deneylerde arşiv büyüyebilir; Drive kaydı varsa orijinal klasör de kalıcıdır. Kaggle'da **Save Version** yapıp Output bölümünü sakla. Colab'da aşağıdaki isteğe bağlı indirme hücresini çalıştır.

Notebook'un kendisini de `.ipynb` olarak indirip public GitHub deposuna ekle. Veri ve büyük checkpoint arşivlerini Git'e yükleme; README'de ayrı bir paylaşım bağlantısı kullan.
""")
code(r"""
import json
import shutil
import zipfile
from datetime import datetime, timezone

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "config_name": CONFIG_NAME, "modes": MODES, "seeds": SEEDS,
    "protocol_frozen": PROTOCOL_FROZEN, "final_strong_eval": FINAL_STRONG_EVAL,
    "synthetic": SYNTHETIC,
    "note": "Smoke is a pipeline check, not a scientific result.",
    "session_training_seconds": training_wall_seconds,
}
(ARTIFACT_ROOT / "notebook_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
archive_path = BASE_DIR / f"madry_mnist_{CONFIG_NAME}_artifacts.zip"
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for root, prefix in [(PROJECT_DIR, "project"), (ARTIFACT_ROOT, "artifacts")]:
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
                continue
            if "data" in p.relative_to(root).parts:  # MNIST yeniden indirilebilir
                continue
            if p.resolve() == archive_path.resolve():
                continue
            archive.write(p, f"{prefix}/{p.relative_to(root).as_posix()}")
print("Arşiv hazır:", archive_path, f"({archive_path.stat().st_size / 1024**2:.1f} MB)")
if IS_COLAB:
    print("İndirmek için sonraki hücrede DOWNLOAD_ARCHIVE=True yapın.")
elif IS_KAGGLE:
    print("Kaggle: Save Version → Output dosyalarını saklayın.")
else:
    from IPython.display import FileLink
    display(FileLink(str(archive_path)))
""")
code(r"""
DOWNLOAD_ARCHIVE = False
if DOWNLOAD_ARCHIVE and IS_COLAB:
    from google.colab import files
    files.download(str(archive_path))
""")
md("""
## Sonraki adımı veriye göre seçelim

- **Smoke tamamlandıysa:** cihaz/config/tablo ekran görüntüsünü ve arşivi sakla; pilot normal eğitimle başla.
- **Pilot tamamlandıysa:** öğrenme eğrilerini ve süreyi karşılaştır; main bütçesini kararlaştır, seed listesini sabitle.
- **Üç seed'li main eğitimleri tamamlandıysa:** ayarları dondur, bütün checkpoint'leri aynı PGD100×20 ile test et.
- **Gerçek sonuçlar hazırsa:** seed tablosu, ortalama ± std, eğitim maliyeti ve makaleden farkları raporla. Gerçek ders LaTeX şablonunu kullan; public repo bağlantısını ekle.

Sunumda her grup üyesi şu zinciri açıklayabilsin: **tehdit modeli → min–max → PGD → adversarial training → güçlü değerlendirme → sonuç ve sınırlılıklar**.
""")
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"name": NOTEBOOK.name, "provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for i, cell in enumerate(cells):
    cell["id"] = f"madry-{i:03d}"
NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK} ({len(cells)} cells, {NOTEBOOK.stat().st_size} bytes)")
