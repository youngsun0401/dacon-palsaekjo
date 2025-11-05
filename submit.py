# 제출에 필요한 파일들을 submit/ 폴더로 내보내고 이를 ZIP파일로 만든다.
# 터미널에서 다음을 실행한다.
# python submit.py

import json
import os
import shutil
from pathlib import Path
import zipfile

def ipynb_to_py(ipynb_path, py_path=None):
    """
    ipynb --> py
    """
    ipynb_path = Path(ipynb_path)
    if not ipynb_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {ipynb_path}")

    # 기본 출력 경로: 같은 이름의 py 파일
    if py_path is None:
        py_path = ipynb_path.with_suffix(".py")

    # ipynb 파일 읽기
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # 코드 셀만 추출
    code_cells = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            code = "".join(cell.get("source", []))
            code_cells.append(code.strip())

    # py 파일로 저장
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(code_cells))

    print(f"파일 추출 완료: `{ipynb_path}` --> `{py_path}`")

def copy_directory(src_dir, dst_dir):
    """
    디렉토리를 통째로 복사 (기존 대상 디렉토리는 삭제)
    """
    def ignore_pycache(dir, names):# 특정 이름 제외
        return {'__pycache__'} if '__pycache__' in names else set()

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if not src_dir.exists():
        print(f"경고: 원본 디렉토리 `{src_dir}` 이 존재하지 않습니다.")
        return

    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    shutil.copytree(src_dir, dst_dir, ignore=ignore_pycache)
    print(f"디렉토리 복사 완료: `{src_dir}` --> `{dst_dir}`")

def copy_file(src_file, dst_file):
    """단일 파일 복사"""
    src_file = Path(src_file)
    dst_file = Path(dst_file)

    if not src_file.exists():
        print(f"경고: 원본 파일 `{src_file}` 이 존재하지 않습니다.")
        return

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    print(f"파일 복사 완료: {src_file} --> {dst_file}")

def make_zip(src_dir, zip_path):
    """디렉토리를 zip 파일로 압축"""
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in src_dir.rglob('*'):
            zipf.write(file, file.relative_to(src_dir))
    print(f"`{zip_path}` 파일 생성.")

if __name__ == "__main__":
    BASE = 'submit'
    os.makedirs(BASE, exist_ok=True)

    ipynb = '2.추론.ipynb'
    py    = BASE + '/script.py'
    ipynb_to_py(ipynb, py)

    copy_directory('model', BASE + '/model')
    copy_directory('mymodules', BASE + '/mymodules')
    copy_file('requirements.txt', BASE + '/requirements.txt')

    make_zip(BASE, f'{BASE}.zip')
