### ▪️ Install Vector Database for Text Data
---

#### Check Docker & Docker Compose

1. Check Docker version

   ```bash
   docker --version
   ```

2. Check Docker Compose version

   ```bash
   docker compose version
   ```

3. Verify permissions

   ```bash
   docker ps
   ```

   * If you see a `permission denied` error, it means your current user is not added to the `docker` group. Run the following commands:
        ```bash
        sudo usermod -aG docker $USER
        newgrp docker
        ```

### ▪️ Install Milvus

---
#### Initial Settings
1. Check directory
    ```
    cd db/milvus_db                      # cd datadrift_dataclinic/db/milvus_db
    ```

2. Remove existing files
    ```
    rm -rf db/milvus_db/docker-compose.yml
    rm -rf db/milvus_db/volumes
    ```
3. Remove containers
    ```
    docker compose down -v

    docker rm -f $(docker ps -aq --filter "name=milvus")
    ```
4. Download the docker-compose.yml file
    ```
    wget https://github.com/milvus-io/milvus/releases/download/v2.3.1/milvus-standalone-docker-compose.yml -O docker-compose.yml
    ```
    - [Additional Troubleshooting] If errors still occur, comment out the version line in the docker-compose.yml file.
        
        ![example-fix_milvus_yaml](img/fix_milvus_yaml.png)

#### Start Milvus
1. Excute Milvus
    ```
    docker compose up -d
    ```

2. Check containers
    ```
    docker ps
    ```
    - Verify that **standalone**, **minio**, and **etcd** are all properly installed and running