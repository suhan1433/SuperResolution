node {
    stage('Clone repository') {
        git branch: 'main', credentialsId: 'github_access_token', url: 'https://github.com/suhan1433/SuperResolution.git'
    }
    stage('Build image') {
        dockerImage = docker.build("suhan123/SuperResolution:v1.0")
    }
    stage('Push image') {
        withDockerRegistry([credentialsId: "docker-access", url: ""]) {
            dockerImage.push()
        }
    }
}

