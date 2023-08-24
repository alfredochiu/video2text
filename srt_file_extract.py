from pathlib import Path
import shutil

# 設定目標檔案夾
target_dir = Path('./finished_srt')

# 如果目標檔案夾不存在，就建立它
if not target_dir.exists():
    target_dir.mkdir()

for file_path in Path('.').rglob('*.srt'):
    dst_path = target_dir / file_path.name
    if dst_path.exists():
        print(f"Skipped {file_path}: {dst_path} already exists")
    else:
        shutil.copy(file_path, dst_path)
        print(f"Copied {file_path} to {dst_path}")
