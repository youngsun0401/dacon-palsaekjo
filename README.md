## 무엇?

DACON - 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회

- [데이터 받는 곳](https://dacon.io/competitions/official/236607/data)
- [A 검사 명세](https://cfiles.dacon.co.kr/competitions/236607/A%EA%B2%80%EC%82%AC(%EC%8B%A0%EA%B7%9C%EA%B2%80%EC%82%AC)%20%EB%AA%85%EC%84%B8.pdf)
- [B 검사 명세](https://cfiles.dacon.co.kr/competitions/236607/B%EA%B2%80%EC%82%AC(%EC%9E%90%EA%B2%A9%EC%9C%A0%EC%A7%80%EA%B2%80%EC%82%AC)%20%EB%AA%85%EC%84%B8.pdf)
- [평가](https://dacon.io/competitions/official/236607/overview/evaluation)
- [공식 제공 코드 - 학습](https://dacon.io/competitions/official/236607/codeshare/13146)
- [공식 제공 코드 - 추론](https://dacon.io/competitions/official/236607/codeshare/13147)





## 작업 방법

### 코드 수정

다음 중 한 방법으로 작업할 수 있다.

- 구글 드라이브에서 공유폴더를 공유받는다.(아래 'Google Drive 공유폴더' 부분 참조)  
  수업에서 처음에 한 것처럼, ipynb 파일을 눌러서 바로 코랩 환경으로 갈 수 있다. 코랩에서 실행 및 코드수정 작업을 한다.
- 공유폴더를 내 컴퓨터에 그대로 복사하여 내 로컬 환경에서 작업한다.



### submit 생성

`create_submit.py` 파일을 실행한다.

```
python create_submit.py
```

실행하면 제출에 필요한 `submit.zip` 파일이 생성된다.





## 프로젝트 관리

※ 주의: 여러 사람이 같은 파일에서 작업하면 실수로 상대방이 작업한 부분을 날리거나, 반대로 내 작업이 날아갈 위험이 있다. 이를 방지하기 위해 다음 중 한 방식으로 작업하자.

- 자신이 작업할 부분을 파이썬 모듈로 분리하여, 그 파일 안에서만 작업한다.
- 프로젝트의 중요 부분을 모두 복붙해서 혼자만의 환경에서 작업한다. 공유폴더에 올릴 때는 다른 폴더나 파일을 덮어쓰지 않게 별도의 폴더(예를 들면 자기 이름으로 된 폴더)에 올린 뒤, 코드관리자에게 취합을 요청한다.

코드관리자: 김영선이 소스코드 관리를 맡는다. 각자 작업 후, 김영선에게 알리면 김영선이 취합하고 깃허브에 올린다.



### Google Drive 공유폴더

1. [공유폴더 링크](https://drive.google.com/drive/folders/1q0Kx5ZGL7Uci7Uh_bifeqtWb30pUNIqm?usp=drive_link)에 들어간다. 이후 내 구글 드라이브의 \[공유 문서함\]에 이 폴더가 생긴다.

2. 그 폴더 우클릭 \> 정리 \> 바로가기 추가

3. 내 구글 드라이브에서 원하는 폴더를 선택한다. 선택된 폴더에 이 공유폴더의 바로가기가 추가된다. 현재 학습, 추론 코드에 적힌 경로는 `/shared-acorn`이다.

4. 이후 코랩에서 구글 드라이브에 마운트하면 이 공유폴더에 접근할 수 있다.

```
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/{ 바로가기 추가에서 선택된 폴더 경로 }/dacon-palsaekjo-shared')
```



### GitHub

[깃허브 링크](https://github.com/youngsun0401/dacon-palsaekjo)

큰 데이터파일은 안 올라갈 뿐 아니라 ipynb 파일은 출력 때문에 git으로 공유하기 번잡하므로 코드 및 데이터 공유 목적은 아니고 간단히 형상관리 간단하게만 할까 함.





## 진행상황

### 이용 패키지

모델은 LightGBM으로 결정했다.



### 맡은 부분

- 김영선: A3, B1, B2 전처리. 소스코드 관리.
- 이혜원: A1, A2, B9, B10 전처리.
- 김용민: A5, A7, B6, B7 전처리.
- 남도욱: A8, A9, B3, B4 전처리.
- 이정우: A4, A6, B5, B8 전처리.



### 소스코드 상태

`1.학습.ipynb`, `2.추론.ipynb`는 baseline의 코드 복붙을 기반으로 수정된 것이다. 바로 실행가능하다.

DACON에서 제공한 학습, 추론 코드가 루트 디렉토리에 있다. 그 두 코드에서 공통 부분을 분리해서 mymodules에 넣었다.

제공된 `open_v2`의 압축을 풀고 프로젝트 루트에 넣는다. 그러나 이 폴더는 gitignore에 포함되어 있다.

학습 데이터 크기가 매우 크므로 git에서 받지 않을 뿐 아니라 간단히 실행해보기도 벅차다. 그래서 `0.copyData.ipynb`를 실행하여 학습 데이터의 앞부분만 따로 저장해 쓴다.(이 잘라낸 데이터를 `data/` 경로에 넣음.) 테스트 데이터는 자르지 않았지만 복사하는 김에 정렬을 더했다.

커밋 충돌을 줄이려고 ipynb 파일은 출력을 지우고 올림.





## 환경 설정

```
conda create -n dacon python=3.10
conda activate dacon
```

`requirements.txt`: 평가 안내 페이지에 나온, 평가 서버에 기본 설치된 패키지의 목록. 다 설치할 필요는 없지만, 여기 있는 라이브러리를 쓸 땐 기왕이면 이 버전에 맞추면 좋을 것이다.





## 평가 방식 정리

[링크](https://dacon.io/competitions/official/236607/overview/evaluation)

우리의 코드 및 필요한 파일들을 `submit.zip`으로 묶어서 제출한다.

`script.py`를 실행하면 우리의 코드가 채점용 입력 데이터가 있는 파일을 읽고, 그에 따른 정답을 예측한 뒤, 이 예측 결과를 파일로 내보내야 한다.

입력 파일의 경로는 제공된 baseline에 나와있는 대로 다음과 같다.

- `data/test.csv`
- `data/test/A.csv`
- `data/test/B.csv`

우리의 예측 결과를 내보낼 파일의 경로는 다음과 같다.

- `output/submission.csv`

로컬 환경에서도 터미털에서 다음을 입력하여 `script.py`를 실행할 수 있다.(※ 아나콘다를 사용할 경우 실행환경 다시 확인!)

```
python script.py
```

혹은, py 파일의 내용 전체를 복사하여 주피터노트북에 갖다붙인 뒤 실행해도 같다.
