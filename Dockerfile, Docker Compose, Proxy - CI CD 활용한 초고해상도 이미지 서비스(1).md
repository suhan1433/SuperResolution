# Dockerfile, Docker Compose, Proxy - CI/CD 활용한 초고해상도 이미지 서비스(3)

---

## 1. Dockerfile

Dockerfile이란?
**Docker 이미지를 생성하기 위한 스크립트 파일이다.** **이미지 빌드 과정에 필요한 명령어와 설정** 포함

이 파일을 기반으로 Docker는 애플리케이션의 실행 환경을 캡슐화한 이미지를 생성합니다.

Dockerfile 이미지는 캡슐 → Container은 이 이미지 캡슐을 풀어 놓은 공간 인스턴스화 한 것 

### 1. 코드 설명

```python
# 베이스 이미지 설정: Python 3.10 slim 같은 불필요한 OS구성 제거하여 경량 이미지를 사용하여 용량을 줄인다. 
FROM python:3.10-slim

# RUN을 통해 이미지 빌드 시 쉘 명령어를 실행하여 설치 가능하다. 
RUN pip install --upgrade pip

# 작업 디렉토리 생성: /web 폴더를 생성
RUN mkdir -p /web

# 환경 변수 설정: APP_PATH에 /web 경로를 저장
ENV APP_PATH /web

# 의존성 파일 복사: requirements.txt를 APP_PATH로 복사
COPY requirements.txt $APP_PATH/

# 의존성 설치: requirements.txt에 정의된 패키지를 설치
# 설치시 생성되는 임시 캐시 파일을 저장하지 않게 하여 이미지 크기를 줄인다. 
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt

# 애플리케이션 파일 복사: app.py와 templates/static 폴더를 APP_PATH로 복사
COPY app.py $APP_PATH/
COPY templates/ $APP_PATH/templates/
COPY static/ $APP_PATH/static/

# 컨테이너 포트 노출: 컨테이너의 5001번 포트를 외부에 노출
EXPOSE 5001

# 컨테이너 시작 명령어: Python으로 /web/app.py 실행
CMD ["python", "/web/app.py"]

```

### 2. Flask

```python
FROM python:3.10-slim

RUN pip install --upgrade pip
RUN mkdir -p /web

ENV APP_PATH /web

COPY requirements.txt $APP_PATH/
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt

COPY app.py $APP_PATH/
COPY templates/ $APP_PATH/templates/
COPY static/ $APP_PATH/static/

EXPOSE 5001

CMD ["python", "/web/app.py"]

```

### 3. ONNX

```python
FROM python:3.10-slim

RUN pip install --upgrade pip
RUN mkdir -p /onnx

ENV APP_PATH /onnx

COPY requirements.txt $APP_PATH/
RUN pip install --no-cache-dir -r $APP_PATH/requirements.txt
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY onnx_server.py $APP_PATH/
COPY inference.py $APP_PATH/
COPY models/ $APP_PATH/models/
COPY utils/ $APP_PATH/utils/
COPY super_resolution.onnx $APP_PATH/

EXPOSE 6000

CMD ["python", "/onnx/onnx_server.py"]

```

### 3. NGINX

Nginx란
**웹 서버**, **리버스 프록시 서버**, **로드 밸런서**, **HTTP 캐시 서버**로 사용된다.

로드 밸런서란 : 여러 대의 서버로 트래픽을 분산하는 것, **고가용성** 갖음 

리버스 프록시란 : 클라이언트 요청을 대신 받아 백엔드에 전달 하고 백엔드 응답을 받아 클라이언트에 전달,  **서버 부하** **분산, 보안** 역할

```docker
# etc는 리눅스 시스템 설정 파일들이 있다. 서비스들 (nginx, mysql 등) 
From nginx:1.21.5-alpine
COPY nginx.conf /etc/nginx/nginx.conf
```

## 2. Docker Compose

Docker Compose란 
여러 개의 Docker 컨테이너를 정의하고 실행할 수 있다.

 특히, **다중 컨테이너 애플리케이션**을 쉽게 설정하고 관리할 수 있게 한다.

### 1. 코드 설명

 

```python
services:  # 서비스들을 정의하는 섹션
  web:  # 'web' 서비스 정의
    image: suhan123/super_resolution:v1.0  # 사용할 Docker 이미지 (super_resolution 이미지의 v1.0 버전)
    deploy:  # 배포 관련 설정
      replicas: 3  # 이 서비스의 복제본 수 (3개의 인스턴스를 실행)
    ports:
      - "5001-5003:5001"  # 호스트의 5001-5003 포트를 컨테이너의 5001 포트로 매핑
    depends_on:  # 이 서비스가 시작되기 전에 실행되어야 하는 다른 서비스들
      - db  # 'db' 서비스가 먼저 실행되어야 함
    networks:
      - app_network  # 이 서비스가 연결될 네트워크 (app_network)

  db:  # 'db' 서비스 정의
    image: mysql:8.0  # 사용할 Docker 이미지 
    restart: always  # 컨테이너가 중지되면 자동으로 재시작
    environment:  # 환경 변수 설정
      MYSQL_ROOT_PASSWORD: root  # 루트 비밀번호 설정
      MYSQL_DATABASE: image_db  # 생성될 기본 데이터베이스 이름
      MYSQL_USER: user  # MySQL 사용자 이름
      MYSQL_PASSWORD: password  # MySQL 사용자 비밀번호
    ports:
      - "23306:3306"  # 호스트의 23306 포트를 컨테이너의 3306 포트로 매핑
    volumes:
      - db_data:/var/lib/mysql  # 데이터베이스 데이터를 저장할 볼륨 (db_data)
    networks:
      - app_network  # 이 서비스가 연결될 네트워크 (app_network)

  proxy-web:  # 'proxy-web' 서비스 정의 (Nginx 프록시 서버)
    image: suhan123/nginx:v1.0  # 사용할 Docker 이미지 (Nginx 이미지의 v1.0 버전)
    container_name: rolling-server-web  # 컨테이너 이름 설정
    restart: always  # 컨테이너가 중지되면 자동으로 재시작
    depends_on:  # 이 서비스가 시작되기 전에 실행되어야 하는 다른 서비스
      - web  # 'web' 서비스가 먼저 실행되어야 함
    ports:
      - '81:80'  # 호스트의 81 포트를 컨테이너의 80 포트로 매핑
    volumes:
      - ${PWD}/nginx/nginx.conf:/etc/nginx/conf.d  # 호스트의 nginx.conf 파일을 컨테이너에 마운트
    networks:
      - app_network  # 이 서비스가 연결될 네트워크 (app_network)

networks:  # Docker 네트워크 정의
  app_network:  # 'app_network' 네트워크 정의
    driver: bridge  # 네트워크 드라이버 설정 (bridge 네트워크 사용)
    ipam:  # IP 주소 관리 설정
      driver: default  # 기본 드라이버 사용
      config:
        - subnet: 172.20.0.0/24  # 서브넷 설정
          ip_range: 172.20.0.0/24  # IP 범위 설정
          gateway: 172.20.0.1  # 게이트웨이 설정

volumes:  # Docker 볼륨 정의
  db_data:  # 'db_data'라는 이름의 볼륨 정의 (MySQL 데이터 지속성을 위한 볼륨)

```

- bridge network란
    - 컨테이너가 가상의 네트워크로 연결되어 있어 같은 네트워크끼리 통신을 가능하게 한다.
    - 포트를 통해 외부와 연결된다: 포트 포워딩. (외부에서 특정 포트로 접근한 요청을 내부 시스템의 다른 포트나 장치로 라우팅)
- subnet이란?
    - IP네트워크를 더 작은 단위로 나눈것.
    - 따로 나누면 각각 다른 정책들을 적용 가능하다
- ip_range란?
    - 서브넷 내에서 할당 가능한 IP 범위
    - 충돌을 방지하기 위해 사용
- gateway란?
    - 외부 네트워크와 연결하기 위한 라우터 역할을 하는 IP주소
    - Docker 네트워크 내의 컨테이너가 **외부 네트워크**(예: 인터넷)와 통신하는 통로 역할을 한다. 이를 통해 컨테이너들이 외부와 연결될 수 있다,
- volume이란?
    - 컨테이너 간 데이터 공유와 지속적인 데이터 저장을 위한 외부 저장소.

### 2. docker-compose.yml

```python
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
    volumes:
      - db_data:/var/lib/mysql
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
    driver: bridge  
    ipam:
      driver: default
      config:
      - subnet: 172.20.0.0/24
        ip_range: 172.20.0.0/24
        gateway: 172.20.0.1

volumes:
  db_data:

```

## 3. Proxy, Load Balancing

리버스 프록시란 : 클라이언트 요청을 대신 받아 백엔드에 전달 하고 백엔드 응답을 받아 클라이언트에 전달,  **서버 부하** **분산, 보안** 역할

로드 밸런서란 : 클라이언트 요청을 대신 받아 3개의 백엔드에 전달과 백엔드에서 받은 것을 다시 3개의 ONNX 모델 서빙 컨테이너에 전달하여 트래픽을 분산 시켜, 고가용성을 얻게함 

### 1. 코드 설명:

라운드 로빈 방법 사용 : 여러 서버에 균등하게 분배하는 방법

단점 : 각 서버의 성능 차이를 고려하지 않는다. 

```python
events { worker_connections 1024; } # 최대 연결 개수를 설정 
http{ # http 요청을 처리하는 부분 
    upstream onnx {  # 로드벨런서를 설정 하는 부분, onnx는 로드벨런서 이름 설정 
    server onnx:8095; # 아래의 세개의 서버로 라운드 로빈 방법으로 트래픽 분산 
    server onnx:8096;
    server onnx:8097;
    }
    server { # 클라이언트 요청을 처리하는 방법 정의 부분 
            listen *:6000 default_server; #nginx가 모든 6000포트에 대해서 수신 대기하도록 함 
    location / { # 요청된 URL에 대해서 어떻게 처리할지 
            proxy_pass http://onnx; # 클라이언트 요청을 onnx 그룹으로 전달 --> onnx그룹은 3개의 서버로 라운드로빈 
            }
    }
}
```

 

### 2. Onnx.conf

```docker
events { worker_connections 1024; }
http{
    upstream onnx { 
    server onnx:8095;
    server onnx:8096;
    server onnx:8097;
    }
    server {
            listen *:6000 default_server; 
    location / {
            proxy_pass http://onnx; 
            }
    }
}
```

### 3. Flask.conf

```docker
events {
   worker_connections 100;
}
http{
   upstream resolution_back {
   server web:5001;
   server web:5002;
   server web:5003;
   }
   server {
	  listen 80  default_server;
   location / {
	  proxy_pass http://resolution_back;
   	}	
   }
}   
```
