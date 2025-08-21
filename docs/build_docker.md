### ▪️ Install Docker 
---

Version 26.1.3 ~ 28.3.3

* WSL 환경에서 이를 설치하려면 추가 설정이 필요
    ```
    wsl --set-default-version 2
    ```
----

1. Update package list
    ```bash
    sudo apt-get update
    ```

2. Install essential packages
    ```bash
    sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common -y
    ```

3. Add Docker GPG key
    ```bash
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    ```

4. Register Docker repository
    ```bash
    sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
    ```

5. Check available Docker versions
    ```bash
    apt-cache madison docker-ce | grep -E "(26\.|27\.|28\.)"
    ```

6. Install specific Docker version (26.1.3 ~ 28.3.3)
    ```bash
    # 예시: Docker 28.3.3 설치 (권장)
    sudo apt-get update
    sudo apt-get install docker-ce=5:28.3.3-1~ubuntu.$(lsb_release -cs) docker-ce-cli=5:28.3.3-1~ubuntu.$(lsb_release -cs) containerd.io -y
    
    # 또는 Docker 27.x 버전
    # sudo apt-get install docker-ce=5:27.3.1-1~ubuntu.$(lsb_release -cs) docker-ce-cli=5:27.3.1-1~ubuntu.$(lsb_release -cs) containerd.io -y
    
    # 또는 Docker 26.x 버전
    # sudo apt-get install docker-ce=5:26.1.3-1~ubuntu.$(lsb_release -cs) docker-ce-cli=5:26.1.3-1~ubuntu.$(lsb_release -cs) containerd.io -y
    ```

7. Prevent Docker auto-update
    ```bash
    sudo apt-mark hold docker-ce docker-ce-cli containerd.io
    ```

### ▪️ Install Docker Compose (Compatible version)
---

```bash
sudo curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### ▪️ Check Installation
---
```bash
docker --version       
docker compose version  
```

### ▪️ Post-installation 
---
Add user to docker group
```bash
# (option) if you don't have the docker group,
sudo groupadd docker

# add user
sudo usermod -aG docker $USER
newgrp docker
```