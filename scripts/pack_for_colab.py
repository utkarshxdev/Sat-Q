"""
scripts/pack_for_colab.py
─────────────────────────
Packs the entire SatQuery AI project into a clean zip file
for easy drag-and-drop upload into Google Colab.

Usage:
    python scripts/pack_for_colab.py
"""
import shutil
import zipfile
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
output_zip = root_dir / "satquery_colab_bundle.zip"

include_dirs = ["satquery", "scripts", "notebooks"]
include_files = ["requirements.txt", "conftest.py", "app.py", "setup.py"]

print(f"Creating Colab bundle: {output_zip.name} ...")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for d_name in include_dirs:
        d_path = root_dir / d_name
        if d_path.exists():
            for f in d_path.rglob("*"):
                if f.is_file() and "__pycache__" not in f.parts and not f.name.endswith(".pyc"):
                    arcname = f.relative_to(root_dir)
                    zipf.write(f, arcname)
                    print(f"  + {arcname}")

    for f_name in include_files:
        f_path = root_dir / f_name
        if f_path.exists():
            zipf.write(f_path, f_name)
            print(f"  + {f_name}")

print(f"\n✓ Successfully created {output_zip} ({output_zip.stat().st_size / 1024:.1f} KB)")
print("👉 You can now drag-and-drop 'satquery_colab_bundle.zip' directly into Google Colab!")
