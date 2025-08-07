### ▪️ Install Ollama

---

Ubuntu에서 Ollama 설치

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

- (option) GPU 설치
    1. NVIDIA GPU 드라이버 설치
        
        ```bash
        nvidia-smi  # 드라이버가 설치되어 있어야 함
        ```
        
    2. CUDA Toolkit 설치
        
        ```bash
        sudo apt install -y nvidia-cuda-toolkit
        ```
        
    3. nividia-container-toolkit 설치
        
        ```bash
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt update
        sudo apt install -y nvidia-container-toolkit
        ```
        
    4. Docker 데몬에 설정 적용
        
        ```bash
        sudo systemctl restart docker
        ```

### ▪️ Initial Settings

---

- Ollama 서비스 시작
    
    ```bash
    sudo systemctl start ollama
    ```
    
- 서비스 상태 확인
    
    ```bash
    systemctl status ollama # active (running) 상태여야 함
    ```