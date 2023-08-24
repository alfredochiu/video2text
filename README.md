# video2text

# 必備檔案

gcp key- \*\*\*\*.json

# 執行檔案

speech.py

## 開始前需要完成的步驟

1. create virtual env `python scripts/create_venv.py`
2. relaunch terminal
3. initial virtual `python scripts/initial_venv.py`

## 加入套件

1. 把套件直接加入到 requirements.txt 當中

## 整理出使用套件 -選擇性

1. 使用 pipreqs
2. `pip install pipreqs`
3. `pipreqs --force --encoding==utf8 --ignore .venv,.mypy_cache,assets ./`
