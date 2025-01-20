---
title: Rustup
date: 2025-01-19 22:57:51
category:
    - Rust
tags:
    - Rust
---

## Install Rust

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Switch between nightly and release version of Rust

```shell
rustup install nightly
rustup default nightly
rustc --version
```