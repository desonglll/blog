---
title: Git Usage
date: 2025-01-19 23:25:25
tags:
    - Git
---


## 查看用户名和邮箱地址

```bash
git config user.name
git config user.password
git config user.email
```

## 配置用户名和邮箱地址

```bash
git config --global user.name "your_name" && git config --global user.password "your_password" && git config --global user.email "your_email"
```

## Configure SSH

```bash
ssh-keygen -t rsa -C "your_email"
```

上述操作执行完毕后，在 `~/.ssh/` 目录会生成 `XXX-rsa` (私钥)和 `XXX-rsa.pub` (公钥)

### add public key to your github

```shell
cat ~/.ssh/id_rsa.pub
```

### add

`Settings` -> `SSH and GPG keys` -> `New SSH key` -> `Add SSH key`

### take a look

```shell
ssh -T git@github.com
```

```shell
Hi desonglll! You've successfully authenticated, but GitHub does not provide shell access.
```

### all in one

```bash
git config --global user.name "username" && git config --global user.password "password" && git config --global user.email "email" && ssh-keygen -t ed25519 -C "email" && cat ~/.ssh/id_ed25519.pub
```

Enabling SSH connections over HTTPS can solve this problem

If you are able to SSH into [git@ssh.github.com](mailto:git@ssh.github.com) over port 443, you can override your SSH settings to force any connection to GitHub.com to run through that server and port.

To set this in your SSH configuration file, edit the file at `~/.ssh/config`, and add this section:

```rust
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```

```shell
git config --global init.defaultBranch main
```

## Git Operations

### 修改 git 的 remote url

如果之前添加的是 `HTTPS` 协议的 github 仓库地址，那么每次 push 或者 pull 仍然需要密码，所以，我们需要将其修改为 `ssh` 协议的，这样，就不需要这么麻烦了。

那么我们应该怎么办呢？

### 查看当前的 remote url

```
 git remote -v
```

```
输出： origin <https://github.com/danygitgit/document-library.git> (fetch)
输出： origin <https://github.com/danygitgit/document-library.git> (push)
```

如果是以上的结果那么说明此项目是使用 `https` 协议进行访问的（如果地址是 git 开头则表示是 `git` 协议）

### 复制远程仓库的 ssh 链接

登陆你的远程仓库，在上面可以看到你的 ssh 协议相应的 url，类似：

`git@github.com:desonglll/codes.git`

复制此 ssh 链接。

### 修改 git 的 remote url

```
git remote origin set-url [url]
```

```
git remote rm origin
```

```
git remote add origin [url]
```

### 合并远程分支到本地

```
git merge origin
```

### 切换分支

```bash
git checkout -b xxx
git branch -M main
```

切换到上一个分支
```bash
git checkout -
```

> git checkout xxx 是指切换到 xxx（用 local 区的 xxx 替换 disk 区文件），-b 意味着 branch，即创建新分支，这条指令合起来意思是创建并切换到 xxx

### 查看文件差异

```bash
# 创建暂缓区
git init
# 查看暂存区与disk区文件的差异
git diff

```

### 添加文件到暂缓区

```bash
# 将xxx文件添加到暂存区
git add xxx

# 将暂存区内容添加到local区的当前分支中
git commit -m "update"

# 将local区的LocalBranchName分支推送到RemoteHostName主机的同名分支
# （若加-f表示无视本地与远程分支的差异强行push）
git push <RemoteHostName> <LocalBranchName>

# 同上，不过改成从远程主机下载远程分支并与本地同名分支合并
git pull <RemoteHostName> <RemoteBranchName>
```

```bash
git rebase xxx
```

> 假设当前分支与 xxx 分支存在共同部分 common，该指令用 xxx 分支包括 common 在内的整体替换当前分支的 common 部分（原先 xxx 分支内容为 common->diversityA，当前分支内容为 common->diversityB，执行完该指令后当前分支内容为 common->diversityA->diversityB）

```bash
# 不加-D表示创建新local分支xxx，加-D表示强制删除local分支xxx
git branch -D xxx
```

### macOS 的.gitignore 文件

在`.gitignore`文件中针对 macOS，您可以添加以下规则来忽略常见的 macOS 生成的文件和文件夹：

```
# macOS system files
.DS_Store
.AppleDouble
.LSOverride

# Icon must end with two \\r
Icon

# Thumbnails
._*

# Files that might appear on external disk
.Spotlight-V100
.Trashes

# Directories potentially created on remote AFP share
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk
```

### 显示 Git 仓库的所有更改历史

要显示 Git 仓库的所有更改历史，您可以使用`git log`命令。`git log`会按照提交的时间顺序列出所有的提交历史记录。

以下是使用`git log`命令的基本用法：

```bash
git log
```

这将显示完整的提交历史记录，包括每个提交的哈希值、作者、提交日期和提交消息。

`git log`还有一些有用的选项可以帮助您更详细地查看历史记录。例如，您可以使用`--oneline`选项以精简的单行格式显示每个提交：

```bash
git log --oneline
```

您还可以使用`--graph`选项以图形化的方式展示提交历史，并显示分支和合并的路径：

```bash
git log --graph
```

### 在 Git 中删除分支

要在 Git 中删除分支，您可以使用`git branch`命令的`-d`或`-D`选项。下面是删除分支的步骤：

1. 首先，确保您不在要删除的分支上工作。您可以通过使用`git branch`命令查看当前分支以及其他分支的列表。

2. 运行以下命令来删除分支：

- 对于已经合并到主分支或其他分支的分支，可以使用`d`选项：
在`<branch-name>`中替换为要删除的分支的名称。

```bash
git branch -d <branch-name>
```

- 对于尚未合并的分支，如果要强制删除分支，可以使用`D`选项：
同样，在`<branch-name>`中替换为要删除的分支的名称。

```bash
git branch -D <branch-name>
```

3. 运行命令后，Git 会删除指定的分支。