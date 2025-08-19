### ▪️ Install Docker
---

1. Update Packages
    ```
    sudo apt-get update
    ```

2. Install essential packages
    ```
    # 필수 패키지 설치
    sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common -y
    ```

3. Add Docker GPG key
    ```
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    ```

4. Register Docker repository
    ```
    sudo add-apt-repository \
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) \
    stable"
    ```
5. Insatll Docker Engine 
    ```
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io -y
    ```

### ▪️ Install Docker Compose
---

```
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### ▪️ Check Docker & Docker Compose
---
```
docker --version
docker-compose --version
```
