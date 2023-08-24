import subprocess


def generate_requirements_txt():
    try:
        # 使用 subprocess 執行 pipreqs 命令
        subprocess.run(["pipreqs", "--force","--encoding==utf8","--ignore" ,".venv","--use-local", "./"])

        print("Requirements generated successfully!")
    except Exception as error:
        print(f"An error occurred: {error}")
