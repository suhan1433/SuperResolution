# SuperResolution Project

<img width="2242" height="1078" alt="image" src="https://github.com/user-attachments/assets/3a05cb62-a6a7-40e3-ab44-8e16bc50e840" />

**목표** : 

의료 이미지의 세밀한 포인트를 더 세밀하게 파악할 수 있도록 하기 위해 개발.

이를 CI/CD 구조로 코드 초고해상도 이미지를 사용자에게 전달하는 것을 목표로 함.

**순서**: 

1. 수정한 코드를 GitHub에 push할 때 webhook을 통해 event가 발생 시킨다
2. webhook을 통해 이벤트가 발생한 것을  알고 Jenkins에서 일련의 과정 진행
    1. github 레파지토리 클론
    2. 각 파일(Flask, Nginx, ONNX)을  빌드
    3. 빌드한 이미지를  Docker hub(public 저장소)에 push
    4. Docker hub에서 Pull을 통해 이미지들을 받는다
    5. Compose Up을 통해 이미지들이 RUN이 되는지 확인 후 배포

[Flask 구성하기 - CI/CD 활용한 초고해상도 이미지 서비스(1)](Flask%20%E1%84%80%E1%85%AE%E1%84%89%E1%85%A5%E1%86%BC%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5%20-%20CI%20CD%20%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%8E%E1%85%A9%E1%84%80%E1%85%A9%E1%84%92%E1%85%A2%E1%84%89%E1%85%A1%E1%86%BC%E1%84%83%E1%85%A9%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%2015e040773032800e851fcc2d79a4166e.md)

[ONNX 구성하기 - CI/CD 활용한 초고해상도 이미지 서비스(2)](ONNX%20%E1%84%80%E1%85%AE%E1%84%89%E1%85%A5%E1%86%BC%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5%20-%20CI%20CD%20%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%8E%E1%85%A9%E1%84%80%E1%85%A9%E1%84%92%E1%85%A2%E1%84%89%E1%85%A1%E1%86%BC%E1%84%83%E1%85%A9%20%E1%84%8B%E1%85%B5%E1%84%86%E1%85%B5%E1%84%8C%2015e040773032807a8be6cb7ea22890e8.md)

[Dockerfile, Docker Compose, Proxy - CI/CD 활용한 초고해상도 이미지 서비스(3)](Dockerfile,%20Docker%20Compose,%20Proxy%20-%20CI%20CD%20%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%2015e040773032802b8457c650a276d659.md)

[Jenkins활용하기 - CI/CD 활용한 초고해상도 이미지 서비스(4)](Jenkins%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%84%80%E1%85%B5%20-%20CI%20CD%20%E1%84%92%E1%85%AA%E1%86%AF%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%8E%E1%85%A9%E1%84%80%E1%85%A9%E1%84%92%E1%85%A2%E1%84%89%E1%85%A1%E1%86%BC%E1%84%83%E1%85%A9%20%E1%84%8B%E1%85%B5%2015e04077303280d0b282f4db7cf630b5.md)

[발생한 문제점 및 추후 계획 ](%E1%84%87%E1%85%A1%E1%86%AF%E1%84%89%E1%85%A2%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A6%E1%84%8C%E1%85%A5%E1%86%B7%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%8E%E1%85%AE%E1%84%92%E1%85%AE%20%E1%84%80%E1%85%A8%E1%84%92%E1%85%AC%E1%86%A8%2015f0407730328084a4ffe866247797ba.md)

결과: 

<img width="1014" height="606" alt="image" src="https://github.com/user-attachments/assets/675abeb2-c25d-48c8-aeeb-19b62ab98d4c" />

<img width="1020" height="1542" alt="image" src="https://github.com/user-attachments/assets/786d195c-e73b-49a4-9cb9-406bfe310017" />

