"""
This is a module-level docstring for the 'os' module.

This module provides a way to interact with the operating system,
such as working with files, directories, and environment variables.
"""
import os
import platform
import subprocess
import sys

# 定義Python版本
PYTHON_VERSION = "3.11"
PYTHON_EXECUTABLE = "python"

def create_virtualenv():
    # 確認是否為支援的作業系統
    system = platform.system()
    if system not in ['Windows', 'Darwin']:
        print("目前只支援 Windows 和 macOS 作業系統。")
        sys.exit(1)

    # 檢查是否已安裝指定版本的Python
    try:
        if f'{sys.version_info.major}.{sys.version_info.minor}' == PYTHON_VERSION:
            print(f"Python 版本為 {PYTHON_VERSION}")
    except subprocess.CalledProcessError:
        print(f"Python {PYTHON_VERSION} 未安裝，請先安裝此版本。")
        sys.exit(1)

    # 建立虛擬環境目錄
    venv_name = ".venv"  # 自定義虛擬環境名稱
    venv_dir = os.path.join(os.getcwd(), venv_name)

    if os.path.exists(venv_dir):
        print(f"目錄 {venv_name} 已存在，請使用其他名稱或刪除現有目錄。")
        sys.exit(1)

    # 使用 venv 模塊建立虛擬環境
    try:
        subprocess.check_output([PYTHON_EXECUTABLE, "-m", "venv", venv_dir])
    except subprocess.CalledProcessError as error:
        print(f"建立虛擬環境時發生錯誤：{error}")
        sys.exit(1)


    # 顯示成功訊息
    print(f"已建立並啟用 Python {PYTHON_VERSION} 虛擬環境，名稱為 {venv_name}。")


if __name__ == "__main__":
    create_virtualenv()