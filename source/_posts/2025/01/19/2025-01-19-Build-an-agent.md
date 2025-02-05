---
title: Build an agent
date: 2025-01-19 22:12:45
tags:
---

## Install Xinference

Reference: [Xinference Docker Image](https://inference.readthedocs.io/en/latest/getting_started/using_docker_image.html)

```shell
docker run -e XINFERENCE_MODEL_SRC=modelscope -p 9998:9997 --gpus all xprobe/xinference:v<your_version> xinference-local -H 0.0.0.0 --log-level debug
```

## Install Dify

Reference: [Deploy with Docker Compose](https://docs.dify.ai/getting-started/install-self-hosted/docker-compose)

### Clone Dify

Clone the Dify source code to your local machine:

```
git clone https://github.com/langgenius/dify.git
```

### Starting Dify

1. Navigate to the Docker directory in the Dify source code

```
cd dify/docker
```

2. Copy the environment configuration file

```
cp .env.example .env
```

3. Start the Docker containers

Choose the appropriate command to start the containers based on the Docker Compose version on your system. You can use the `$ docker compose version` command to check the version, and refer to the [Docker documentation](https://docs.docker.com/compose/install/) for more information:

- If you have Docker Compose V2, use the following command:

```
docker compose up -d
```

- If you have Docker Compose V1, use the following command:

```
docker-compose up -d
```

After executing the command, you should see output similar to the following, showing the status and port mappings of all containers:

```
[+] Running 11/11
 ✔ Network docker_ssrf_proxy_network  Created                                                                 0.1s 
 ✔ Network docker_default             Created                                                                 0.0s 
 ✔ Container docker-redis-1           Started                                                                 2.4s 
 ✔ Container docker-ssrf_proxy-1      Started                                                                 2.8s 
 ✔ Container docker-sandbox-1         Started                                                                 2.7s 
 ✔ Container docker-web-1             Started                                                                 2.7s 
 ✔ Container docker-weaviate-1        Started                                                                 2.4s 
 ✔ Container docker-db-1              Started                                                                 2.7s 
 ✔ Container docker-api-1             Started                                                                 6.5s 
 ✔ Container docker-worker-1          Started                                                                 6.4s 
 ✔ Container docker-nginx-1           Started                                                                 7.1s
```

Finally, check if all containers are running successfully:

```
docker compose ps
```

This includes 3 core services: `api / worker / web`, and 6 dependent components: `weaviate / db / redis / nginx / ssrf_proxy / sandbox` .

```
NAME                  IMAGE                              COMMAND                   SERVICE      CREATED              STATUS                        PORTS
docker-api-1          langgenius/dify-api:0.6.13         "/bin/bash /entrypoi…"   api          About a minute ago   Up About a minute             5001/tcp
docker-db-1           postgres:15-alpine                 "docker-entrypoint.s…"   db           About a minute ago   Up About a minute (healthy)   5432/tcp
docker-nginx-1        nginx:latest                       "sh -c 'cp /docker-e…"   nginx        About a minute ago   Up About a minute             0.0.0.0:80->80/tcp, :::80->80/tcp, 0.0.0.0:443->443/tcp, :::443->443/tcp
docker-redis-1        redis:6-alpine                     "docker-entrypoint.s…"   redis        About a minute ago   Up About a minute (healthy)   6379/tcp
docker-sandbox-1      langgenius/dify-sandbox:0.2.1      "/main"                   sandbox      About a minute ago   Up About a minute             
docker-ssrf_proxy-1   ubuntu/squid:latest                "sh -c 'cp /docker-e…"   ssrf_proxy   About a minute ago   Up About a minute             3128/tcp
docker-weaviate-1     semitechnologies/weaviate:1.19.0   "/bin/weaviate --hos…"   weaviate     About a minute ago   Up About a minute             
docker-web-1          langgenius/dify-web:0.6.13         "/bin/sh ./entrypoin…"   web          About a minute ago   Up About a minute             3000/tcp
docker-worker-1       langgenius/dify-api:0.6.13         "/bin/bash /entrypoi…"   worker       About a minute ago   Up About a minute             5001/tcp
```

With these steps, you should be able to install Dify successfully.

### Upgrade Dify

Enter the docker directory of the dify source code and execute the following commands:

```
cd dify/docker
docker compose down
git pull origin main
docker compose pull
docker compose up -d
```

#### Sync Environment Variable Configuration (Important)

- If the `.env.example` file has been updated, be sure to modify your local `.env` file accordingly.
- Check and modify the configuration items in the `.env` file as needed to ensure they match your actual environment. You may need to add any new variables from `.env.example` to your `.env` file, and update any values that have changed.

### Access Dify

Access administrator initialization page to set up the admin account:

```
# Local environment
http://localhost/install

# Server environment
http://your_server_ip/install
```

Dify web interface address:

```
# Local environment
http://localhost

# Server environment
http://your_server_ip
```