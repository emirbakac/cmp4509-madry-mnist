"""Build the standalone recorded-results analysis notebook (no model execution)."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / 'notebooks/02_verify_recorded_results.ipynb'
EMBEDDED_FILES = (
    'results/FINAL_TEST_ALL_MODELS.csv',
    'results/FINAL_TEST_PROTOCOL.json',
    'results/validation_reported.json',
    'tools/verify_results.py',
)


def build_notebook():
    cells = []

    def add(kind, source):
        source = textwrap.dedent(source).strip() + '\n'
        cell = {'cell_type': kind, 'id': hashlib.sha256((str(len(cells)) + source).encode()).hexdigest()[:12],
                'metadata': {}, 'source': source}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(source):
        add('markdown', source)

    def code(source):
        add('code', source)

    embedded = {}
    bundle = hashlib.sha256()
    for relative in EMBEDDED_FILES:
        raw = (PROJECT / relative).read_bytes()
        bundle.update(relative.encode() + b'\0' + raw + b'\0')
        embedded[relative] = {'sha256': hashlib.sha256(raw).hexdigest(),
                              'base64': base64.b64encode(raw).decode('ascii')}

    md('''
    # CMP4509 — Kayıtlı MNIST sonuçlarını denetle ve görselleri yeniden üret

    Bu notebook, **önceden kaydedilmiş dokuz modelin sonuç tablosunu** okuyarak
    ortalama ± örnek standart sapmayı ve rapor şekillerini yeniden üretir.
    **Model eğitmez, checkpoint yüklemez, MNIST indirmez ve saldırı çalıştırmaz.**
    Aritmetik/biçim denetimi, tarihî deneyin bağımsız olarak yeniden yürütülmesi anlamına gelmez.

    Notebook gereken dört dosyayı kendi içinde taşır; depo klonlamak veya GPU kullanmak gerekmez.
    Colab, Kaggle veya yerel Jupyter'da hücreleri sırayla çalıştırın.
    Yalnız eksik NumPy/Matplotlib paketlerinin ilk kurulumu internet gerektirebilir.

    - Ana test: dokuz kayıt, üç yöntem × üç eğitim seed'i, kayıt başına 10.000 MNIST test görüntüsü.
    - Belgelenen saldırı: L∞, ε=0.3, α=0.01, PGD100 × **5** rastgele başlangıç.
      Bu ayarlar özgün CSV'nin sütunları değildir; ayrı protokol kaydından gelir.
    - Ek validation: bildirilen aynı 512 görüntü için PGD100 × **20** ve full standard AutoAttack.
      Yüzdeler yazarın yuvarlatılmış beyanıdır; ham loglar ve subset indeksleri mevcut değildir.
      Test ile validation sonuçları birleştirilmez.
    - Checkpoint SHA-256 alanlarının biçimi denetlenir; model dosyaları bulunmadığından
      hash'lerin gerçek checkpoint içerikleriyle eşleşmesi doğrulanamaz.

    Yeni eğitim/saldırı deneyleri için ayrı `01_mnist_reproduction.ipynb` notebook'unu kullanın.
    Public proje: https://github.com/emirbakac/cmp4509-madry-mnist
    ''')
    md('''
    ## 1 · Yalnız eksik analiz paketlerini kur

    Kurulu paketler korunur. Bu analiz için PyTorch, torchvision veya AutoAttack gerekmez.
    ''')
    code('''
    import importlib.util
    import subprocess
    import sys

    for module, requirement in {
        "numpy": "numpy>=1.26,<3",
        "matplotlib": "matplotlib>=3.8,<4",
    }.items():
        if importlib.util.find_spec(module) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

    print("Python:", sys.version.split()[0])
    print("Analiz ortamı hazır; GPU gerekmiyor.")
    ''')
    md('''
    ## 2 · Gömülü kaynakların birebir kopyasını ayrı analiz klasörüne aç

    Çalışma dizini platforma göre `/kaggle/working`, `/content` veya mevcut klasör altında
    oluşturulur. Her gömülü kaynak sürümünün içerik hash'inden türeyen ayrı bir snapshot'ı vardır.
    Aynı sürümü yeniden çalıştırmak kaynakları değiştirmez. Mevcut snapshot dosyası elle
    değiştirilmişse işlem durur; kullanıcı dosyalarının üzerine sessizce yazılmaz.
    Hesaplanan istatistik ve şekiller ise aynı snapshot içinde yeniden oluşturulabilir.
    ''')
    code('''
    from pathlib import Path
    import base64
    import hashlib
    import json
    import os

    if "_RECORDED_ANALYSIS_BASE" not in globals():
        _RECORDED_ANALYSIS_BASE = (
            Path("/kaggle/working") if Path("/kaggle/working").is_dir()
            else Path("/content") if Path("/content").is_dir()
            else Path.cwd()
        )
    ANALYSIS_ROOT = _RECORDED_ANALYSIS_BASE / "madry_mnist_recorded_analysis"
    ''')
    code('EMBEDDED = ' + json.dumps(embedded, ensure_ascii=False, sort_keys=True, indent=2)
         + '\nBUNDLE_SHA256 = ' + repr(bundle.hexdigest()))
    code('''
    SNAPSHOT_DIR = ANALYSIS_ROOT / ("snapshot_" + BUNDLE_SHA256[:16])
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    for relative, item in EMBEDDED.items():
        raw = base64.b64decode(item["base64"], validate=True)
        assert hashlib.sha256(raw).hexdigest() == item["sha256"], relative
        destination = SNAPSHOT_DIR / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != raw:
            raise RuntimeError(
                f"Mevcut kaynak değişmiş: {destination}. Dosya korunuyor; "
                "ANALYSIS_ROOT için yeni bir klasör seçip tekrar çalıştırın."
            )
        if not destination.exists():
            destination.write_bytes(raw)
    print("Kaynak snapshot:", SNAPSHOT_DIR)
    print("Kaynak paketi SHA-256:", BUNDLE_SHA256)
    print("Dört gömülü kaynağın byte/hash denetimi geçti.")
    ''')
    md('''
    ## 3 · Kayıtları ve protokolün kanıt sınırlarını göster

    CSV değerleri **0–1 aralığında oran** olarak saklanır; aşağıdaki kaynak tablosunda da
    bu biçim korunur. Örneğin `0.975`, **%97.5** demektir; `975` değildir.
    Tarihî eğitim veya saldırı tekrarını bu hücrelerden çıkarmayın.
    ''')
    code('''
    import csv
    from IPython.display import Markdown, display

    protocol = json.loads((SNAPSHOT_DIR / "results/FINAL_TEST_PROTOCOL.json").read_text())
    validation = json.loads((SNAPSHOT_DIR / "results/validation_reported.json").read_text())
    with (SNAPSHOT_DIR / "results/FINAL_TEST_ALL_MODELS.csv").open(newline="") as handle:
        recorded = list(csv.DictReader(handle))
    columns = ["method", "seed", "best_epoch", "clean_accuracy", "fgsm_accuracy", "pgd_accuracy", "test_images"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    table += ["| " + " | ".join(row[column] for column in columns) + " |" for row in recorded]
    display(Markdown("\\n".join(table)))
    print("Test protokolünün kaynağı:", protocol["protocol_provenance"])
    print("Validation kaynağı:", validation["provenance"])
    print("Validation durumu:", validation["status"])
    ''')
    md('''
    ## 4 · İstatistikleri ve yayın için şekilleri yeniden hesapla

    Denetim; özgün CSV hash'ini, dokuz satırı, her yöntem için seed 0/1/2'yi,
    seçili epoch'ları, 10.000 görüntülük paydaları, oran aralıklarını ve hash biçimini kontrol eder.
    Ortalama ve **örnek standart sapma (ddof=1)** eğitim seed'leri üzerinden hesaplanır;
    bu standart hata veya güven aralığı değildir. İki düşük doğruluklu PGD40 koşusu korunur.

    Çıktılar `results/verified/` ve `report/figures/` altında oluşturulur.
    Validation özetleri, raporlanmış yuvarlatılmış yüzdelerden hesaplanır. AutoAttack için
    %83.34 ortalama bu girdilere aittir; mevcut olmayan kesin sayımlar üretilmez.
    ''')
    code('''
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(SNAPSHOT_DIR / ".matplotlib_cache"))
    completed = subprocess.run(
        [sys.executable, str(SNAPSHOT_DIR / "tools/verify_results.py")],
        cwd=SNAPSHOT_DIR, env=environment, text=True, capture_output=True,
    )
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr)
        completed.check_returncode()
    VERIFIED_DIR = SNAPSHOT_DIR / "results/verified"
    (VERIFIED_DIR / "analysis_log.txt").write_text(completed.stdout + completed.stderr)
    display(Markdown((VERIFIED_DIR / "SUMMARY.md").read_text()))
    statistics = json.loads((VERIFIED_DIR / "statistics.json").read_text())
    assert statistics["n_records"] == 9 and statistics["n_seeds_each"] == 3
    print("Aritmetik ve kaynak biçimi denetimi tamamlandı. Eğitim/saldırı yeniden çalıştırılmadı.")
    ''')
    md('''
    ## 5 · Şekilleri incele

    İlk iki şekil kaydedilmiş **10.000 görüntülük test** sonuçlarını gösterir.
    Son şekil, ham logları bulunmayan **512 görüntülük validation beyanlarını** gösterir;
    genişletilmiş ölçek açıkça etiketlenir. Seed'ler bağımsız eğitim koşularıdır, zaman adımları değildir.
    PNG ve vektörel PDF sürümleri birlikte kaydedilir.

    Rapor/sunum için tablo ve şekillerin ekran görüntülerini de saklayın; asıl sayısal kaynak
    CSV/JSON dosyalarıdır. Bu şekiller sertifikalı dayanıklılık kanıtı değildir.
    ''')
    code('''
    from IPython.display import Image

    FIGURE_DIR = SNAPSHOT_DIR / "report/figures"
    for stem in ("final_accuracy_comparison", "seed_robustness", "validation_attack_comparison"):
        display(Markdown("### " + stem.replace("_", " ")))
        display(Image(filename=str(FIGURE_DIR / (stem + ".png"))))
    ''')
    md('''
    ## 6 · Analiz çıktılarının ZIP kopyasını sakla

    ZIP; gömülü kaynak kopyalarını, yeniden hesaplanan istatistikleri ve PNG/PDF şekillerini içerir.
    **Checkpoint veya yeni saldırı sonucu içermez.** Kaggle'da dosyaları kalıcı tutmak için
    **Save Version** kullanın. Colab otomatik indirmesi isteğe bağlıdır; varsayılan kapalıdır.
    ''')
    code('''
    import zipfile
    from IPython.display import FileLink

    archive = SNAPSHOT_DIR / "recorded_results_analysis.zip"
    paths = sorted(
        path for folder in (SNAPSHOT_DIR / "results", FIGURE_DIR, SNAPSHOT_DIR / "tools")
        for path in folder.rglob("*") if path.is_file()
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in paths:
            entry = zipfile.ZipInfo(path.relative_to(SNAPSHOT_DIR).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            package.writestr(entry, path.read_bytes())
    print("Çıktı arşivi:", archive)
    display(FileLink(os.path.relpath(archive, Path.cwd())))

    DOWNLOAD_ZIP = False
    if DOWNLOAD_ZIP:
        try:
            from google.colab import files
        except ImportError:
            print("Otomatik indirme yalnız Colab içindir; üstteki bağlantıyı kullanın.")
        else:
            files.download(str(archive))
    ''')
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3'},
            'recorded_results_bundle_sha256': bundle.hexdigest(),
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Notebook destination; --check inspects this path without writing.')
    parser.add_argument('--check', action='store_true', help='Fail if the notebook is absent or out of sync.')
    args = parser.parse_args()
    generated = json.dumps(build_notebook(), ensure_ascii=False, indent=2) + '\n'
    if args.check:
        if not args.output.exists() or args.output.read_text() != generated:
            raise SystemExit(f'Notebook is missing or out of sync: {args.output}')
        print(f'Notebook synchronized: {args.output}')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated)
        print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
