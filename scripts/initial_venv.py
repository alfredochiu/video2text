import os
import platform
import subprocess


def read_requirements(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


def get_installed_packages():
    try:
        result = subprocess.run(
            ["python", "-m", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        installed_packages = result.stdout.splitlines()[2:]
        return [package.split()[0].lower() for package in installed_packages]
    except subprocess.CalledProcessError as error:
        print(f"Error: {error}")
        return []


def check_packages(required_packages):
    installed_packages = get_installed_packages()
    missing_packages = [
        package
        for package in required_packages
        if package.lower() not in installed_packages
    ]

    if missing_packages:
        print("Some packages are missing:")
        for package in missing_packages:
            print(f"  {package}")
    else:
        print("All required packages are installed.")


def initial_venv():
    system = platform.system()
    venv_name = ".venv"  # 自定義虛擬環境名稱
    venv_dir = os.path.join(os.getcwd(), venv_name)

    # 啟用虛擬環境
    activate_script = "Scripts" if system == "Windows" else "bin"
    if system == ['Darwin', 'Linux']:
        activate_path = os.path.join(venv_dir, activate_script, "activate")
        activate_cmd = f"source {activate_path}"
    else:
        activate_path = os.path.join(venv_dir, activate_script, "Activate")
        activate_cmd = '& ' + activate_path + '.ps1'

    print("啟用虛擬環境...")
    subprocess.call(activate_cmd, shell=True)

    # 安裝必要的套件（這裡可以根據你的需求自行增減）
    print("安裝必要的套件...")
    subprocess.call(
        ["python", "-m", "pip", "install", "-r", "requirements.txt"]
    )

    # 檢查是否所有的 requirements.txt 套件都已經安裝完成
    required_packages = read_requirements("requirements.txt")
    check_packages(required_packages)


if __name__ == "__main__":
    initial_venv()
