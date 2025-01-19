---
title: Rust Rules
date: 2025-01-19 21:35:10
tags:
    - Rust
---

## 2.10 The Three Causes of Data Races

1. Two or more pointers access the same data at the same time.
2. At least one of the pointers is being used to write to the data.
3. There’s no mechanism being used to synchronize access to the data.

## 2.11 The Three Rules of Ownership
1. Each value in Rust has a variable that’s called its owner.
2. There can only be one owner at a time.
3. When the owner goes out of scope, the value will be dropped.

## 2.12 The Two Rules of References
1. At any given time, you can have either one mutable reference or any number of immutable references.
2. References must always be valid.