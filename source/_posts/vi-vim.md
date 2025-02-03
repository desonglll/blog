---
title: vi&vim
tags:
  - null
category:
  - null
date: 2025-01-26 20:33:21
---

## Install neovim on Ubuntu

https://launchpad.net/~neovim-ppa/+archive/ubuntu/stable

https://launchpad.net/~neovim-ppa/+archive/ubuntu/unstable

## Adding this PPA to system

```shell
sudo add-apt-repository ppa:neovim-ppa/unstable
sudo apt update
sudo apt install neovim -y
```

## Save your buffer with the ex command

Today, when a 500-gigabyte drive is considered small, errors like this are generally rare. If something like this does occur, you have several courses of action to take:  

1. **Save Data to a Different File System**  
   - Try to write your file somewhere safe on a different file system (such as `/tmp`) to save your data.  

2. **Force Buffer Preservation**  
   - Use the Ex command `:pre` (short for `:preserve`) to force the system to save your buffer.  

3. **Free Up Disk Space**  
   - If the above methods don’t work, look for files to remove:  
     - Open a graphical file manager (e.g., Nautilus on GNU/Linux) to find and remove old files you no longer need.  
     - Use `CTRL-Z` to suspend `vi` and return to the shell prompt. From there, you can use Unix commands to locate large files:  
       - `df`: Displays how much disk space is free on a given file system or the system as a whole.  
       - `du`: Shows how many disk blocks are used for specific files and directories.  
         - Example: `du -s * | sort -nr` lists files and directories sorted by space usage in descending order.  
     - Once done removing files, use `fg` to bring `vi` back to the foreground and save your work normally.  

4. **Use Shell Commands from Within `vi`**  
   - Besides using `CTRL-Z` and job control, you can type `:sh` to start a new shell.  
     - Use `CTRL-D` or `exit` to terminate the shell and return to `vi`.  
     - This even works with `gvim`!  
   - Alternatively, you can use `:!` to run shell commands directly from `vi`. For example:  
     ```bash
     :!du -s *
     ```  
     This runs the command and then returns to editing once the command is done.