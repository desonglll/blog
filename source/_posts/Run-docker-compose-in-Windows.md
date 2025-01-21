---
title: Run docker-compose in Windows
tags:
  - Docker
  - Docker Compose
category:
  - Docker
date: 2025-01-21 17:36:48
---


## Install Docker

[Download Docker for your machine](https://www.docker.com/products/docker-desktop/) 

## Install WSL2

右键Windows图标，打开 PowerShell（以管理员身份运行）

```powershell
wsl --install
wsl --set-default-version 2
```

## Docker Compose file

Create a `docker-compose.yml` file in a folder.

Then paste the following configuration into `docker-compose.yml`

```yml
# docker-compose.yml
services:
  backend:
    image: desonglll/sd-get-backend:latest
    container_name: sd-get-backend
    ports:
      - "8000:8000"
    environment:
      - DJANGO_SETTINGS_MODULE=sd_pro.settings
      - DEBUG=1

  frontend:
    image: desonglll/sd-get-frontend:latest
    container_name: sd-get-frontend
    ports:
      - "3000:80"
```

Windows + R enter `cmd` and press enter.

Enter the folder where `docker-compose.yml` located in using `cd` command.

Finally, press

```shell
docker-compose up -d
```

## Open entrypoint

```
http://localhost:3000/sd-pro/
```