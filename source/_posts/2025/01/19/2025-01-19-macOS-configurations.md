---
title: macOS configurations
date: 2025-01-19 23:03:24
tags:
    - macOS
---

## Set Control+Command+mouse/trackpad moves the window

```shell
defaults write -g NSWindowShouldDragOnGesture -bool true
```

After opening, hold down cmd+control to drag the window to any position.

## Fix destroyed

```shell
sudo xattr -r -d com.apple.quarantine xxx.app
```

```shell
#!/bin/bash
clear
BLACK="\\033[0;30m"
DARK_GRAY="\\033[1;30m"
BLUE="\\033[0;34m"
LIGHT_BLUE="\\033[1;34m"
GREEN="\\033[0;32m"
LIGHT_GREEN="\\033[1;32m"
CYAN="\\033[0;36m"
LIGHT_CYAN="\\033[1;36m"
RED="\\033[0;31m"
LIGHT_RED="\\033[1;31m"
PURPLE="\\033[0;35m"
LIGHT_PURPLE="\\033[1;35m"
BROWN="\\033[0;33m"
YELLOW="\\033[0;33m"
LIGHT_GRAY="\\033[0;37m"
WHITE="\\033[1;37m"
NC="\\033[0m"

echo ""
echo ""
echo -e "${LIGHT_CYAN}MACYY-www.macyy.cn${NC} - Mac破解软件分享中心"

parentPath=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parentPath"
appPath=$( find "$parentPath" -name '*.app' -maxdepth 1)
appName=${appPath##*/}
appBashName=${appName// /\\ }
appDIR="/Applications/${appBashName}"
echo ""
echo ""
echo -e "ℹ️  『${appBashName} 已损坏，无法打开/ 来自身份不明的开发者』等问题无法打开软件修复实用工具"
echo ""
#未安装APP时提醒安装，已安装绕过公证
if [ ! -d "$appDIR" ];then
  echo ""
  echo -e "⚠️  执行结果：${RED}您还未安装 ${appBashName} ，请先按照安装教程安装${NC}"
  else
  #绕过公证
  echo -e "ℹ️  ${YELLOW}请输入开机密码，输入完成后按下回车键（输入过程中密码是看不见的，如果有任何问题可以添加群聊联系我们：www.macyy.cn/faq/archives/127 ）${NC}"
  sudo spctl --master-disable
  sudo xattr -rd com.apple.quarantine /Applications/"$appBashName"
  echo ""
  echo ""
  echo -e "✅  执行结果：${GREEN}修复成功！${NC}您现在可以正常运行 ${appBashName} 了。更多Mac破解软件尽在MacYY${BLUE}www.macyy.cn${NC}"
fi
echo ""
echo ""
echo -e "操作已完成✅"
echo -e "本窗口可以关闭！"
```

## 允许安装任何来源

```
sudo spctl --master-disable
```

## 设置 VSCode 按键重复

```
defaults write -g ApplePressAndHoldEnabled -bool false

```

## Install Homebrew

```
/bin/bash -c "$(curl -fsSL <https://cdn.jsdelivr.net/gh/ineo6/homebrew-install/install.sh>)"
```

## 查看端口被哪个程序占用

- 1.查看端口被哪个程序占用
    - `sudo lsof -i tcp:port`
    - 如：`sudo lsof -i tcp:8082`
- 2.看到进程的 PID，可以将进程杀死。
    - `sudo kill -9 PID`
    - 如：`sudo kill -9 3210`

## 关闭门禁

```
sudo spctl --master-disable
```

## 清除网络代理

```
networksetup -setwebproxystate Wi-Fi off
networksetup -setsecurewebproxystate Wi-Fi off
networksetup -setsocksfirewallproxystate Wi-Fi off
```

## Accelerate mouse move speed

```shell
defaults write NSGlobalDomain KeyRepeat -int 1
defaults write NSGlobalDomain InitialKeyRepeat -int 15
```

- 第一行的 `KeyRepeat` 对应的是「按键重复」，系统设置里调到最快对应的值是 `2`，你可以调成 `0` 或者 `1`（建议调为 `1`，`0` 可能会太快）；
- 第二行的 `InitialKeyRepeat` 对应的是「重复前延迟」，系统设置里调到最快对应的值是 `15`，你可以尝试调成 `10` 或者更小，不过我还是建议保持 `15`，因为反应时间太快会容易导致误操作（比如 Esc 键和 Command-Z 这样的快捷键）；

你也许会注意到系统设置采用的措辞是「重复」，而不是我描述的「移动」，是因为这些选项也适用于字符输入，比如长按字母 `a` 会输出一堆「aaaaaaaaa」。但是重复输入字符这个功能在有些电脑上是被禁用的，你需要在「终端」应用输入以下代码来开启（需重启电脑）：

```shell
defaults write NSGlobalDomain ApplePressAndHoldEnabled -bool false
```

VScode/Sublime在mac下vim的长按hjkl键无法持续移动光标

终端下执行命令：

```shell
defaults write com.microsoft.VSCode ApplePressAndHoldEnabled -bool false
```

最后重启vscode即可

若要复原，执行命令:

```shell
defaults write com.microsoft.VSCode ApplePressAndHoldEnabled -bool true
```

对于其他app也是同样的方法，只需要把`com.microsoft.VSCode`改成对于app的ID即可

如:设置Sublime `defaults write com.sublimetext.3 ApplePressAndHoldEnabled -bool false`

## Set proxy for npm

```
查看当前源地址：
npm config get registry

切换源地址命令如下：

切换至淘宝源：
npm config set registry=http://registry.npm.taobao.org/

切换至npm源：
npm config set registry=http://registry.npmjs.org

临时使用：
npm --registry <https://registry.npm.taobao.org> install express
```

## `tree` command parameters

- a 显示所有文件和目录。
- A 使用 ASNI 绘图字符显示树状图而非以 ASCII 字符组合。
- C 在文件和目录清单加上色彩，便于区分各种类型。
- d 显示目录名称而非内容。
- D 列出文件或目录的更改时间。
- f 在每个文件或目录之前，显示完整的相对路径名称。
- F 在执行文件，目录，Socket，符号连接，管道名称名称，各自加上`` `/` `=` `@` `|`号。
- g 列出文件或目录的所属群组名称，没有对应的名称时，则显示群组识别码。
- i 不以阶梯状列出文件或目录名称。
- I 不显示符合范本样式的文件或目录名称。
- l 如遇到性质为符号连接的目录，直接列出该连接所指向的原始目录。
- n 不在文件和目录清单加上色彩。
- N 直接列出文件和目录名称，包括控制字符。
- p 列出权限标示。
- P 只显示符合范本样式的文件或目录名称。
- q 用`?`号取代控制字符，列出文件和目录名称。
- s 列出文件或目录大小。
- t 用文件和目录的更改时间排序。
- u 列出文件或目录的拥有者名称，没有对应的名称时，则显示用户识别码。
- x 将范围局限在现行的文件系统中，若指定目录下的某些子目录，其存放于另一个文件系统上，则将该子目录予以排除在寻找范围外。

当然你也可以直接通过`tree --help`查询

### 查看不同级别子目录和文件

使用`tree -L N`这个命令，只查看当前第 N 级的目录和文件
使用`tree -L 1`这个命令，只查看当前第一级的目录
使用`tree -L 2`这个命令，只查看当前第二级的目录和文件

