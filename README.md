## 무엇?

DACON - 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회

- [데이터 받는 곳](https://dacon.io/competitions/official/236607/data)
- [A 검사 명세](https://cfiles.dacon.co.kr/competitions/236607/A%EA%B2%80%EC%82%AC(%EC%8B%A0%EA%B7%9C%EA%B2%80%EC%82%AC)%20%EB%AA%85%EC%84%B8.pdf)
- [B 검사 명세](https://cfiles.dacon.co.kr/competitions/236607/B%EA%B2%80%EC%82%AC(%EC%9E%90%EA%B2%A9%EC%9C%A0%EC%A7%80%EA%B2%80%EC%82%AC)%20%EB%AA%85%EC%84%B8.pdf)
- [평가](https://dacon.io/competitions/official/236607/overview/evaluation)
- [공식 제공 코드 - 학습](https://dacon.io/competitions/official/236607/codeshare/13146)
- [공식 제공 코드 - 추론](https://dacon.io/competitions/official/236607/codeshare/13147)





## 소스코드 현상황

DACON에서 제공한 학습, 추론 코드가 루트 디렉토리에 있다. 그 두 코드에서 공통 부분을 분리해서 mymodules에 넣었다.

제공된 `open_v2`의 압축을 풀고 프로젝트 루트에 넣는다. 그러나 이 폴더는 gitignore에 포함되어 있다.

학습 데이터 크기가 매우 크므로 git에서 받지 않을 뿐 아니라 간단히 실행해보기도 벅차다. 그래서 `0.copyData.ipynb`를 실행하여 별도 디렉토리(`open_dev`)에 학습 데이터의 앞부분만 따로 저장해 쓴다. 테스트 데이터는 자르지 않았지만 복사하는 김에 정렬을 더했다.

커밋 충돌을 줄이려고 ipynb 파일은 출력을 지우고 올림.



## 환경 설정

```
conda create -n dacon python=3.10
conda activate dacon
```

`requirements.txt`: 평가 안내 페이지에 나온, 평가 서버에 기본 설치된 패키지의 목록.

```
pip install -r requirements.txt
```
