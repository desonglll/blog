---
title: Create a Ubuntu container in Docker
date: 2025-01-19 21:35:10
tags:
    - Docker
    - Ubuntu
---

## Run a ubuntu with docker

### Create a Ubuntu container

```shell
docker run -d --name my-ububntu -p 22:22 -p 8080:8080 ubuntu # <host_port:container_port>
docker ps -a
docker exec -it <container_id> /bin/bash
```

### Install `openssh-server`

```shell
apt update
apt install vim openssh-server -y
passwd root # Setting a password for root
vim /etc/ssh/sshd_config
# Change `PermitRootLogin yes`
service ssh restart
service ssh start
```

## [How to automatically start ssh server when launching Windows Subsystem for Linux](https://gist.github.com/dentechy/de2be62b55cfd234681921d5a8b6be11)
