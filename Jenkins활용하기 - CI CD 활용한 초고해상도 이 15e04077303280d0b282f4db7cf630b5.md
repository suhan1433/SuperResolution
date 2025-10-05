# Jenkins활용하기 - CI/CD 활용한 초고해상도 이미지 서비스(4)

---

## 1. CI/CD

**CI (Continuous Integration)**:

- 변경된 코드를 지정된 저장소(예: Docker Hub, Github)에 정기적으로 통합하고, 자동화된 빌드 및 테스트를 실행하여 코드 품질을 지속적으로 확인하는 프로세스.
- 목표는 코드 변경 사항을 자주 통합하여, 통합 후 발생할 수 있는 오류를 조기에 발견하는 것.

**CD (Continuous Delivery / Continuous Deployment)**:

- **Continuous Delivery**: 코드 변경이 자동으로 빌드되고 테스트되어, 배포 가능한 상태로 준비되며, 운영 환경에 배포하기 위한 마지막 단계까지 자동화되는 프로세스.
- **Continuous Deployment**: 코드가 자동으로 운영 환경에 배포까지 완료되는 프로세스.

**CI/CD**는 개발 프로세스에서 **자동화된 빌드, 테스트, 배포**를 통해 **품질 향상**과 **릴리스 주기 단축**을 목표로 하는 방법론입니다.

# 2. JenKins 사용

```python
docker pull jenkins/jenkins:lts # jenkins lts버전을 다운
docker inspect [image명] # 포트 확인 

# 외부 에서 18080으로 들어오면 컨테이너로 8080포트로 연결하도록 설정  
# 여러 노드에 jenkins 구성 시 50000으로 구성 
docker run -itd --name=fc-jenkins -p 18080:8080 -p 50000:50000 \ 
--privileged=true \ # 컨테이너 내부 주요 자원 접근 가능 권한 부여
-u root \ # 여러 권한 문제 및 명령어 수행 위해
--restart=always \ # 
# jenkins에서 docker build 사용을 위해 host에 설치된 docker.sock을 같이 사용하기 위해 볼륨 구성
# jenkins안에 도커를 설치하기 위해 사용 --> 그래야 jenkins가 build가 가능하다 
-v /var/run/docker.sock:/var/run/docker.sock \  
# Jenkins home과 애플리케이션 소스가 제공될 영역을 공유해야한다
#  /home/kevin/fastcampus/jenkins 여기에 제공을 하면 jenkins_home에 공유가 되어 이곳에서 build 진행 
-v /home/kevin/fastcampus/jenkins:/var/jenkins_home \ 
jenkins/jenkins:lts

sudo netstat -nltp | grep docker-proxy #열려있는 것을 확인 
```

<img width="464" height="316" alt="image" src="https://github.com/user-attachments/assets/6799b8db-3e86-4fd5-a646-4f09bef96d38" />


docker inspect로 이미지 포트 확인 

<img width="1880" height="824" alt="image" src="https://github.com/user-attachments/assets/24c5cf1c-8f3f-4666-8230-35aa883f0219" />


```python
docker exec fc-jenkins cat [위 위치로] 

docker exec -it -u root fc-jenkins bash

mkdir .ssh && cd $_

# 키 이름 설정 
ssh-keygen -t rsa-f /var/jenkins_home/.ssh/fc-jenkins

cat /var/jenkins_home/.ssh/fc-jenkins # private key 
cat /var/jenkins_home/.ssh/fc-jenkins.pub #public key 
```

**Private Key :**  

- **Jenkins 안에서 인증을 위해 사용**
- **ssh 접근에 사용**

**Public Key:**  

- **깃헙의 ssh 키값으로 사용**

## 1. 인증

### 1. 시스템 인증

jenkins관리 → crdentials 클릭 → global add credential 클릭  → ssh, private key 입력 후 생성 

<img width="1472" height="776" alt="image" src="https://github.com/user-attachments/assets/d98d012f-6148-4d42-9502-c31eec5c1e6c" />


<img width="1436" height="1488" alt="image" src="https://github.com/user-attachments/assets/473f58e4-2ae9-4eb0-a7e3-99b26238d5fa" />


### 2. github 인증

settings → deploy keys → Add new → public key 입력   

<img width="1584" height="1292" alt="image" src="https://github.com/user-attachments/assets/0445c42f-8096-49d8-82b5-21b3c8a3caf2" />


- github와 jenkins 연결을 위해

github 계정의 settings → developer Settings → personal access tokens → tokens (classic)→ jenkins 인증 추가→ username(깃헙), password(token) 

<img width="2242" height="444" alt="image" src="https://github.com/user-attachments/assets/1ab2fce5-a790-4dd7-abe1-a1306d4ffde3" />


select scope

workflow, **admin:repo_hook, gist, user,  delete_repo 선택** 

<img width="1096" height="1338" alt="image" src="https://github.com/user-attachments/assets/9e9fbf26-499b-453a-9492-80792d84be84" />


### 3. docker hub인증

Account settings → Security → personal access token → 토큰 생성 

<img width="1206" height="886" alt="image" src="https://github.com/user-attachments/assets/53ae2ce8-f8cb-4b7e-bc11-075d18b88f2d" />


- 도커와 jenkins 연결을 위해

jenkins관리 → crdentials 클릭 → global add credential 클릭  → username(docker hub이름), password(token) 

<img width="910" height="1500" alt="image" src="https://github.com/user-attachments/assets/a99ddc73-ccee-4fe6-a7dd-88e2f7fb028c" />


## 2. webhook 설정

webhooks란?

특정 이벤트(git push, commit 등) 서비스나 응용프로그램에 알림 제공 

ec2 기반이면 public ip를 사용하면 되지만 현재 프로젝트는 vm을 사용하기에 외부에서 접근할 수 있는 public ip가 없기에 ngrok을 활용하여 만든다. 

- github과 jenkins에 연결 해야한다.

```python
sudo snap ngrok # https://download.ngrok.com/linux?tab=snap
ngrok config add-authtoken [] # 가입 후 토큰 입력 
ngrok http 18080 # jenkins포트인 18080로 연결 하여 webhook에 사용할 public ip 사용 

```

<img width="1356" height="922" alt="image" src="https://github.com/user-attachments/assets/f68bc596-5e58-483b-8ff5-2da3004f7082" />


<img width="1622" height="488" alt="image" src="https://github.com/user-attachments/assets/dd4c919d-b2c5-49e7-9394-8a3a7e18242f" />


- public 주소 …free.app 복사

### 1.  github webhook

settings → webhooks → 추가 

<img width="1712" height="870" alt="image" src="https://github.com/user-attachments/assets/ee5a643d-ba5b-4260-8214-aef5696f926d" />


- github webhook에 추가

### 2. Jenkins webhook

Jenkins 관리 → system → Jenkins Location  URL 변경 

<img width="856" height="462" alt="image" src="https://github.com/user-attachments/assets/e2392f74-4290-42fa-bf35-1c99d073f456" />


## 3. Jenkins Plugin

Jenkins가 hostOS의 컨테이너에서 돌아가기에 SSH가 필요 

jenkins 관리 → plugin → ssh, docker pipeline 설치 

- Plugins 설치 목록
    - **public OverSSH** #
    - **Docker Pipeline** #
    - **Generic Webhook Trigger**  #웹 훅에 대해 trigger 하는 것
    - **GitHub integration** # 깃헙 연결을 위해

<img width="2726" height="788" alt="image" src="https://github.com/user-attachments/assets/bb85fb56-250a-4c89-ad92-273387f79e6b" />


## 4.  Jenkins 사용하여 코드 배포하기

### 1.  jenkins 컨테이너에  Docker 설치

컨테이너 안의 jenkins에 접속하여 docker을 설치 해야한다. → jenkins로 build 및 컨테이너 관리 진행 위해 

```python
docker exec -it -u root fc-jenkins bash # 접속 
# 설치
curl https://get.docker.com/ > dockerinstall && chmod 777 dockerinstall && ./dockerinstall
# 확인
docker ps # jenkins 컨테이너도 동일한 도커 환경 구성 

```

- Jenkins에서 docker ps 결과 host os에서의 결과가 같은 이유?
    - jenkins안에 도커를 설치하면 jenkins 설치 시 docker.sock을 볼륨으로 공유하기에 docker ps 결과 host os와 jenkins 컨테이너 안에서의 결과가 같게 나온다.

### 2.  Git 설정

공유하려는 코드 폴더에 git 설치 

```python
# /home/kevin/fastcampus/jenkins 설정한 볼륨에 들어가서 소스코드 넣고 
git init
ls -al # .git 폴더 확인
sudo git config --global user.email "josuhan1433@gmail.com"
sudo git config --global user.name "suhan1433"
sudo git remote add origin https://github.com/suhan1433/breeds-classification.git #origin 경로 설정
git remote -v # 현제 저장소 확인 
git add .
git commit -m 'test'
git push -u origin main # main 앞에 +을 붙이면 강제 
  
```

### 3. Jenkins 파이프라인

Dashboard → 새로운 Item 만들기 → Pipline 선택 

<img width="1244" height="1422" alt="image" src="https://github.com/user-attachments/assets/238524ef-cb45-45ba-abec-687d29b3bfce" />


General에서 → Git Project 선택(git 주소) → Build Trigger에서 → GitHub hook trigger for GITScm polling 선택 ( github에 webhook이 왔을 때) → Pipline 설정 형상관리(SCM) 

<img width="1070" height="1016" alt="image" src="https://github.com/user-attachments/assets/c93990f5-6ada-45e2-9916-defd1e9e9f33" />

<img width="646" height="520" alt="image" src="https://github.com/user-attachments/assets/06d1f90c-b90b-43f8-a46a-23e0205360f9" />

<img width="1976" height="1234" alt="image" src="https://github.com/user-attachments/assets/470ae4cc-12c6-49e0-b88a-bcc837e3d5d3" />

<img width="828" height="318" alt="image" src="https://github.com/user-attachments/assets/92d13247-3def-4d9d-a1fa-595cdea72260" />


### 4. Pipeline에 Jenkins 파일 넣기 위해 작성

```python
#처음 설정 한 애플리케이션 수행할 볼륨 경로로 이동 /home/kevin/fastcampus/jenkins
docker compose up -d 
# jenkins 경로에 소스 파일(애플리케이션) 복사 후 이동
cd jenkins/source/ 
vi JenkinsFile # 젠킨스 파일 생성  
```

```python
# JenkinsFile 작성 
# Jenkins 페이지에서 인증서 만들떄 사용한 ID값을 사용한다. 
pipeline {
    agent any
    stages {
        stage('Clone repository') {
            steps {
                git branch: 'main', credentialsId: 'github_access_token', url: 'https://github.com/suhan1433/SuperResolution.git'
            }
        }
        
        stage('Build Flask App image') {
            steps {
                dir('flask_app') { // flask_app 디렉토리로 이동
                    script {
                        dockerImage = docker.build("suhan123/super_resolution:v1.0")
                    }
                }
            }
        }

        stage('Build Nginx image') {
            steps {
                dir('nginx') { // nginx 디렉토리로 이동
                    script {
                        dockerImageNginx = docker.build("suhan123/nginx:v1.0")
                    }
                }
            }
        }

        stage('Build ONNX Server image') {
            steps {
                dir('onnx_server') { // onnx_server 디렉토리로 이동
                    script {
                        dockerImageOnnx = docker.build("suhan123/onnx_server:v1.0")
                    }
                }
            }
        }
        
        stage('Push images') {
            steps {
                withDockerRegistry([credentialsId: 'docker-access', url: '']) {
                    script {
                        // 각 이미지를 Docker Hub에 푸시
                        dockerImage.push()
                        dockerImageNginx.push()
                        dockerImageOnnx.push()
                    }
                }
            }
        }
    }
}

```

### 5.  Pipline을 사용하는 Project 생성

Jenkins dashboard → 새로운 item 만들기 → Freestyle project 선택 

<img width="1892" height="474" alt="image" src="https://github.com/user-attachments/assets/dca2898c-9b91-4058-812e-c91800de3800" />

<img width="994" height="926" alt="image" src="https://github.com/user-attachments/assets/027d409c-5f7f-422d-b5bd-da7af8259733" />

<img width="988" height="1366" alt="image" src="https://github.com/user-attachments/assets/3e305665-ed26-4094-a961-0a395d252d95" />

<img width="744" height="680" alt="image" src="https://github.com/user-attachments/assets/ac787046-5584-4577-a93b-47098e911ca4" />


docker 이미지가 build 된 후 push된 새로운 이미지로 docker compose 수행하여 새로운 컨테이너 생성

```python
# 처음 jenkins 설치 시 볼륨 연결한 곳 
cd /var/jenkins_home/[project명] # jenkins의 source project 위치 
docker compose down # 컨테이너 내리기 
docker pull suhan123/[project명]:v1.0 # 새로운 이미지 docker hub에서 pull 
docker compose up -d # 컨테이너 올리기 

```

<img width="1264" height="656" alt="image" src="https://github.com/user-attachments/assets/e27ebc86-41d2-4416-8503-51d6c8156a76" />


```python
# compose 수정 
# /home/kevin/fastcampus/jenkins/[project]/docker-compose.yml

version: '3.8'

services:
  web:
    image: suhan123/super_resolution:v1.0
    deploy:
      replicas: 3
    ports:
      - "5001-5003:5001"
    depends_on:
      - onnx
      - db
    networks:
      - app_network 

  onnx:
    image: suhan123/onnx_server:v1.0
    deploy:
      replicas: 3
    ports:
      - "8095-8097:6000"
    networks:
      - app_network  

  db:
    image: mysql:8.0
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: image_db
      MYSQL_USER: user
      MYSQL_PASSWORD: password
    ports:
      - "23306:3306"
    networks:
      - app_network 

  proxy-web:
    image: suhan123/nginx:v1.0
    container_name: rolling-server-web
    restart: always
    depends_on:
     - web
    ports:
     - '81:80'
    volumes:
     - ${PWD}/nginx/nginx.conf:/etc/nginx/conf.d
    networks:
     - app_network

  proxy-onnx:
   image: suhan123/nginx:v1.0
   container_name: rolling-onnx-lb
   restart: always
   ports:
    - '6000:80'
   volumes:
    - ${PWD}/nginx/nginx_onnx.conf:/etc/nginx/conf.d
   networks:
    - app_network
 
networks:
  app_network:
    driver: bridge  # web과 db가 연결되는 네트워크
    ipam:
      driver: default
      config:
      - subnet: 172.20.0.0/24
        ip_range: 172.20.0.0/24
        gateway: 172.20.0.1

```
