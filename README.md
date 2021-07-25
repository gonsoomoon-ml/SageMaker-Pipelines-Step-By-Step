# SageMaker-Pipelines-Step-By-Step 워크샵

## 1.워크샵 배경
- SageMaker Pipelines의 Modeling Building Pipepine (MBP) 을 Step-By-Step으로 진행을 하면서 배우는 것에 목적을 가집니다. 여기에 사용된 데이터 세트 및 일부 코드는 여기 블로그에서 가져왔습니다. [Blog: Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo](https://aws.amazon.com/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/). 이 워크샵을 완료후에 이 블로그의 참조를 권장 드립니다.
    - 원본 블로그와 이 워크샵의 차이점은 주어진 비즈니스 문제에 원본은 세이지 메이커 솔류션의 End-To-End로 접근을 합니다. 하지만 이 워크샵은 세이지 메이커의 **"데이터 전처리인 프로세싱" -->  "훈련" -->  "모델 배포 및 추론"** 를 현업에서 실제로 개발 하듯이 Step-By-Step으로 접근합니다. 이 과정속에서 단계에 대한 내용 및 코드 또한 배우게 됩니다.
    

- MBP 의 각 단계(스텝) 별로 노트북이 구성 되어있고, 이 단계를 모두 마무리 한후에 마지막으로 모든 단계를 연결하여 파이프라인 생성을 합니다. 이후 MBP를 스케줄링하여 특정 시간에 실행하는 예시도 포함 합니다.

- 아래는 최종 파이프라인을 연결하여 완료된 MBP 입니다.

![fraud2scratch2pipeline.png](img/fraud2scratch2pipeline.png)

### [알림] 이 워크샵의 다음 과정은 여기를 참조 하세요. --> [고급 단계](phase01/Phase01-README.md)

## 2.비즈니스 문제

<i> "자동차 보험 사기는 보험 신청에 대한 사실을 허위로 표현하고 보험 청구를 부풀리며 사고를 준비하고 한 번도 발생하지 않은 부상 또는 손상에 대한 청구 양식 제출, 도난 차량에 대한 허위 보고에 이르기까지 다양합니다.  IRC(Insurance Research Council)의 조사에 따르면, 보험 사기는 2012 년 자동차 보험 신체 상해에 대한 총 청구 금액의 15~17 %를 차지했습니다.
이 연구는 2012년에 자동차 보험 신체 상해 지불 청구액 중 56억 ~ 77억 달러가 사기로 판명되었다고 추정했으며, 이는 2002년의 43억 ~ 58억 달러 대비 크게 증가하였습니다.</i>" [source: Insurance Information Institute](https://www.iii.org/article/background-on-insurance-fraud)

본 예제에서는 자동차 보험 도메인을 사용하여 사기 가능성이 있는 청구를 감지합니다. 구체적으로 <i>"주어진 자동 청구가 사기 일 가능성은 무엇입니까?"</i> 의 유즈케이스를 다루며, 기술 솔루션을 탐색합니다.


## 3. 데이터 세트 
모델 및 워크플로를 구축하기 위한 입력은 보험 데이터의 두 테이블 인 클레임(cliams) 테이블과 고객(customers) 테이블입니다. 이 데이터는 SageMaker Processing로 전처리를 합니다. (원본 블로그는 SageMaker Data Wrangler 로 전처리를 합니다.)

## 4. 선수 지식 및 기술 사용 컴포넌트

### 4.1. 선수 지식
- AWS Cloud 일반 지식 (S3, EC2, IAM 등) (초급/중급 정도 수준)
- Python 코딩 (Pandas, Numpy 패키지 초급/중급 정도 수준)
- ML 기초 지식 수준  (Jupyter Notebook 사용 경험 있음)
    
### 4.2. 세이지메이커 추천 선수 지식    
- [SageMaker Overview (소개 동영상)](https://youtu.be/jF2BN98KBlg)
- [SageMaker demo](https://youtu.be/miIVGlq6OUk) 
    - (1시간 데모에 많은 내용을 압축해서 다루고 있습니다. 반복해서 보시거나 돌려보기로 차근차근 보셔도 괜찮습니다.)
- [세이지메이커 셀프 스터디](https://github.com/gonsoomoon-ml/Self-Study-On-SageMaker)
    
### 4.3. 기술 사용 컴포넌트    
- 데이터 전처리
    - 세이지 메이커 로컬 모드로 다커 컨테이너에서 전처리 수행
    - MBP의 단일 단계인 step_process 정의 및 실행
- 모델 훈련
    - 로컬 모드 및 호스트 모드로 다커 컨테이너에서 모델 훈련 수행 (SageMaker Script 모드 사용)
    - MBP의 단일 단계인 step_train 정의 및 실행
- 모델 배포 및 추론
    - 로컬 모드에서 다커 컨테이너에서 엔드포인트 생성 
    - MBP의 단일 단계인 step_deploy 정의 및 실행
- MBP 연결
    - 위의 정의된 단계를 모두 연결하여 한개의 MBP 생성 및 실행
- 이벤트 브릿지에 스케줄 등록
    - 1시간 마다 위의 MBP 실행

    

## 5.핸즈온 환경
- [필수] 세이지 메이커의 노트북 인스턴스에서 쥬피터 노트북을 통해 실행하세요.
- [권장] 또한 세이지 메이커의 스튜디오에서 생성된 파이프라인을 GUI 로 보면서 상태를 확인 하세요.

## 6. 노트북 구성

### [Quick Approach] 
- Step-By-Step으로 접근하지 않고, 빠르게 보시기 위해서는 아래 노트북만을 수행 하세요.
    - 2.1.Prepare-Dataset.ipynb
    - 8.5.all-pipeline.ipynb
    - 9.1.Inference-Analyze.ipynb    

### [Step-By-Step 접근]

- 1.1.Explore-Dataset.ipynb
    - 데이터 세트가 어떻게 구성이 되었는지 데이터를 탐색 합니다.


- 2.1.Prepare-Dataset.ipynb
    - 데이터 세트 (claims.csv, customers.csv) 를 S3에 업로드


- 3.1.Preprocess-Pipeline.ipynb
    - 첫번째의 전처리 과정의 스텝 생성 하며 아래 세가지의 단계별 구현을 합니다.
        - 로컬 노트북에서 전처리 스크립트 실행 (예: python preprocess.py)
        - 로컬 노트북에서 로컬 모드의 다커 컨테이너로 실행
        - MBP 의 파이프라인 단계 (step_proecess) 를 생성하여 실행
        
        
-  4.1.Train-Pipeline.ipynb
    - 전처리된 데이터를 바탕으로 모델 훈련을 합니다.
        - (1) 로컬 노트북 인스턴스에서 다커 컨테이너로 훈련 코드 실행 (로컬 모드로 불리움)
        - (2) 세이지 메이커 호스트 모드로 (로컬 노트북 인스턴스에서 실행이 되는 것이 아님) 다커 컨테이너를 통해서 훈련 코드 실행
        - (3) MBP 의 파이프라인 단계 (step_train) 를 생성하여 실행
        
        
- 6.1.Create-Model-Pipeline.ipynb 
    - 세이제 메이커 모델 생성 단계
    
    
- 8.1.Deploy-Pipeline.ipynb
    - 실시간 엔드포인트 생성 단계
        - (1) 로컬 노트북 인스턴스에서 다커 컨테이너로 엔드포인트를 생성
        - (2) 로컬에서 Endpoint 생성 스크립트 실행 (Boto3 API 이용)
        - (3) SageMaker Pipeline 에서 엔드포인트 생성을 수행합니다.


- 8.5.all-pipeline.ipynb
    - 위에 생성한 모든 MBP 스텝을 연결하여 모든 파이프라인을 생성 함.
    
    
- 9.1.Inference-Analyze.ipynb
    - 최종적으로 생성한 실시간 엔드포인트에 추론 테스트
    
    
- 9.5.schedule-pipeline.ipynb
    - 매일 매 시간마다 MBP를 실행하는 예시
    

### [알림] 이 워크샵의 다음 과정은 여기를 참조 하세요. --> [고급 단계](phase01/Phase01-README.md)

## [참고 자료]

- Blog: Architect and build the full machine learning lifecycle with AWS: An end-to-end Amazon SageMaker demo
    - https://aws.amazon.com/blogs/machine-learning/architect-and-build-the-full-machine-learning-lifecycle-with-amazon-sagemaker/
    - Code: https://github.com/aws/amazon-sagemaker-examples/tree/master/end_to_end+
    
- Amazon SageMaker Model Building Pipelines
    - 세이지 메이커 모델 빌딩 파이프라인의 개발자 가이드
    - https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines.html

- Amazon SageMaker Local Mode Examples    
    - 로컬 모드에서 사용하는 예시 (로컬 노트북에서 PyCharm 으로 훈련 및 서빙 예시)
    - https://github.com/aws-samples/amazon-sagemaker-local-mode

- Run Amazon SageMaker Notebook locally with Docker container
    - https://towardsdatascience.com/run-amazon-sagemaker-notebook-locally-with-docker-container-8dcc36d8524a
    
    
- XGBoost Parameters
    - XGBoost 파라미터 설명
    - https://xgboost.readthedocs.io/en/latest/parameter.html
