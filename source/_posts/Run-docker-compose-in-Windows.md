---
title: Run docker-compose in Windows
tags:
  - Docker
  - Docker Compose
category:
  - Docker
date: 2025-01-21 17:36:48
---

## Build with log

```shell
docker build -t <name> . --no-cache --platform linux/amd64 --progress=plain
```

## Build Docker image using Github Action

[Reference](https://medium.com/@wasdsro/tutorial-building-a-docker-container-via-github-actions-8636bdc931b1)

## Install Docker

[Download Docker for your machine](https://www.docker.com/products/docker-desktop/) 

## Install WSL2

Right click Windows icon, open PowerShell(with admin)

```powershell
wsl --install
wsl --set-default-version 2
```

## Docker Compose file

```shell
docker run -d \
  --name sd-pro-backend \
  -p 8000:8000 \
  -e DJANGO_SETTINGS_MODULE=sd_pro.settings \
  -e DEBUG=1 \
  desonglll/sd-pro-backend:latest
```

```shell
docker run -d \
  --name sd-pro-frontend \
  -p 3000:80 \
  desonglll/sd-pro-frontend:latest
```

**ALL IN ONE**

```shell
docker run -d --name sd-pro-backend -p 8000:8000 -e DJANGO_SETTINGS_MODULE=sd_pro.settings -e DEBUG=1 desonglll/sd-pro-backend:latest && \
docker run -d --name sd-pro-frontend -p 3000:80 desonglll/sd-pro-frontend:latest
```

**OR**

Create a `docker-compose.yml` file in a folder.

Then paste the following configuration into `docker-compose.yml`

```yml
# docker-compose.yml
services:
  backend:
    image: desonglll/sd-pro-backend:latest
    container_name: sd-pro-backend
    ports:
      - "8000:8000"
    environment:
      - DJANGO_SETTINGS_MODULE=sd_pro.settings
      - DEBUG=1

  frontend:
    image: desonglll/sd-pro-frontend:latest
    container_name: sd-pro-frontend
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
http://localhost:8000/admin/ # mike 070011
```

![Done](https://scontent-tpe1-1.xx.fbcdn.net/v/t39.30808-6/474221949_1991857497964677_2622951236302317547_n.jpg?stp=cp6_dst-jpg_tt6&_nc_cat=101&ccb=1-7&_nc_sid=f727a1&_nc_ohc=yZqVN7GBwuwQ7kNvgGDjRP6&_nc_zt=23&_nc_ht=scontent-tpe1-1.xx&_nc_gid=AZGcV_nPBoq6UKCLD2Y4FTv&oh=00_AYBbSAKbBWyrmR4UBDZjItAonjcKwEXPb7FhU8FaEqCWtA&oe=679599B5) 

## Github Actions

```yml
name: Build and Push Multi-Arch Docker Images

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}

      - name: Setup QEMU
        uses: docker/setup-qemu-action@v2
        with:
          platforms: all

      - name: Setup Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build and push frontend image
        uses: docker/build-push-action@v4
        with:
          context: ./frontend
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/sd-pro-frontend:latest
          platforms: linux/amd64,linux/arm64

      - name: Build and push backend image
        uses: docker/build-push-action@v4
        with:
          context: ./backend
          push: true
          tags: ${{ secrets.DOCKER_USERNAME }}/sd-pro-backend:latest
          platforms: linux/amd64,linux/arm64
```