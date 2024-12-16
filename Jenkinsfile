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

