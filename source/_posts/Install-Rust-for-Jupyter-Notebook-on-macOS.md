---
title: Install Rust for Jupyter Notebook on macOS
date: 2025-01-19 23:38:19
tags:
    - Rust
    - Jupyter
    - macOS
---

You can install rust for `jupyter` on mac with [Evcxr Jupyter Kernel](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md)

I follow the tutorial from [the video](https://www.youtube.com/watch?v=0UEMn3yUoLo&ab_channel=Dr.ShahinRostami)

1. Install `miniconda/anaconda`
2. Create and activate `conda` environment

```lua
conda create -n rustjupyter python=3.8
conda activate rustjupter
```

3. Install `jupyterlab` (1.1.4), `cmake`, `nodejs` (13.10.1)

```r
conda install -c conda-forge jupyterlab==1.1.4
conda install -c anaconda cmake
conda install -c conda-forge nodejs=13.10.1
```

4. Install `rust` (1.68.0+)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

5. Install `evcxr_jupyter`

```css
cargo install evcxr_jupyter
evcxr_jupyter --install
```

6. Run `jupyter` lab

```
jupyter lab
```

7. Go to `http://localhost:8888/` on browser, select **Rust** in *Notebook* section