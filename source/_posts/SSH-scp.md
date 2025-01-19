---
title: SSH scp
date: 2025-01-19 21:35:10
tags:
   - SSH
   - Ubuntu
---


`scp` 本身不支持直接在命令中指定密码，这是出于安全性的考虑。然而，你可以使用以下几种方法绕过手动输入密码的过程。

------

### 使用 SSH 密钥免密登录

这是推荐的安全方法。

1. **生成 SSH 密钥**（如果尚未生成）：

   ```bash
   ssh-keygen -t rsa -b 4096
   ```

   按提示完成操作，通常会生成 `~/.ssh/id_rsa` 和 `~/.ssh/id_rsa.pub`。

2. **将公钥复制到目标服务器**：

   ```bash
   ssh-copy-id -p 8182 mikeshinoda@192.168.2.3
   ```

   输入密码完成操作。

3. **测试免密登录**：

   ```bash
   ssh -p 8182 mikeshinoda@192.168.2.3
   ```

4. **重新运行 `scp` 命令**（无需输入密码）：

   ```bash
   scp -P 8182 -r ./docs/.vitepress/dist/ mikeshinoda@192.168.2.3:/home/mikeshinoda/web
   ```