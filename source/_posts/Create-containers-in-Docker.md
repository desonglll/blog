---
title: Create containers in Docker
date: 2025-01-19 21:35:10
tags:
    - Docker
    - Ubuntu
category:
    - Docker
---

## Run a ubuntu with docker

### Create a Ubuntu container

```shell
docker run -itd --name my-ubuntu -p 22:22 -p 8080:8080 -p 80:80 -p 5173:5173 ubuntu /bin/bash # <host_port:container_port>
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

### [How to automatically start ssh server when launching Windows Subsystem for Linux](https://gist.github.com/dentechy/de2be62b55cfd234681921d5a8b6be11)


## Create a redis container

[Hub](https://hub.docker.com/_/redis) 

Start a redis instance

```
docker run --name some-redis -d redis
```

Start with persistent storage

```
docker run --name some-redis -d redis redis-server --save 60 1 --loglevel warning
```
