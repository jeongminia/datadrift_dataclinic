### ▪️ Install Ollama

---

#### Install Ollama on Ubuntu

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### (option) GPU Setup
Run these steps only if the installation output does not show
NVIDIA GPU installed.

1. Check NVIDIA GPU driver
    ```
    nvidia-smi                           # Driver must be installed
    ```
    - If the driver version is displayed → skip the following steps.
    - If not, install the driver before proceeding.

2. Install CUDA Toolkit
    ```
    sudo apt install -y nvidia-cuda-toolkit
    ```
        
3. Install NVIDIA Container Toolkit
    ```
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    ```
        
4. Restart Docker daemon
    ```
    sudo systemctl restart docker
    ```

### ▪️ Initial Settings

---

1. Start the Ollama
    ```
    sudo systemctl start ollama
    ```

2. make dir for storing models
    ```
    sudo mkdir -p /usr/share/ollama
    sudo chown -R ollama:ollama /usr/share/ollama
    ```

3. write down
    ```
    sudo -u ollama bash -lc 'touch /usr/share/ollama/.w && rm /usr/share/ollama/.w'
    ```

4. restart and Check service status 
    ```
    sudo systemctl restart ollama
    systemctl status ollama                # Should be active (running)
    ```
