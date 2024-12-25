---
title: Rust Programming Language
editLink: ture
---

<script setup>
import { watch } from "vue";
import { useData } from 'vitepress'

const { isDark } = useData()
watch(
  isDark,
  (newValue) => {
    const link = document.querySelector("link[rel='icon']");
    if (link) {
      link.href = newValue
        ? "./Rust_programming_language_white_logo.svg"
        : "./Rust_programming_language_black_logo.svg";
    }
  },
  { immediate: true }
);
</script>

# Rust

<img
    v-if="isDark"
    src="./Rust_programming_language_white_logo.svg"
    alt="Rust Logo - White"
    style="width: 200px; display: block; margin: 0 auto;"
/>
<img
    v-else
    src="./Rust_programming_language_black_logo.svg"
    alt="Rust Logo - Black"
    style="width: 200px; display: block; margin: 0 auto;"
/>

```rust
fn main() {
    println!("Hello, world!");// [!code focus]
}
```

[[toc]]

## Rust Playground: Try Rust without installing

<Mention text="Rust Playground" href="https://play.rust-lang.org/" from="rust-lang" /> is a web-based tool that allows you to write, run, and share Rust code.

## Installing Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Is Rust up to date?

```bash
rustc --version
rustup update
```

## Cargo: the Rust build tool and package manager

When you install Rustup you'll also get the latest stable version of the Rust build tool and package manager, also known as Cargo. Cargo does lots of things:

- build your project with `cargo build`
- run your project with `cargo run`
- test your project with `cargo test`
- build documentation for your project with `cargo doc`
- publish a library to crates.io with `cargo publish`

To test that you have Rust and Cargo installed, you can run this in your terminal of choice:

```bash
cargo --version
```

## Quick Start

### Generating a New Project

Let’s write a small application with our new Rust development environment. To start, we’ll use Cargo to make a new project for us. In your terminal of choice, run:

```bash
cargo new hello-rust
```

This will generate a new directory called `hello-rust` with the following files:

```
hello-rust
|- Cargo.toml
|- src
  |- main.rs
```

`Cargo.toml` is the manifest file for Rust. It’s where you keep metadata for your project, as well as dependencies.

`src/main.rs` is where we’ll write our application code.

The `cargo new` step generated a "Hello, world!" project for us! We can run this program by moving into the new directory that we made and running this in our terminal:

```bash
cargo run
```

You should see this in your terminal:

```text
$ cargo run
   Compiling hello-rust v0.1.0 (/Users/ag_dubs/rust/hello-rust)
    Finished dev [unoptimized + debuginfo] target(s) in 1.34s
     Running `target/debug/hello-rust`
Hello, world!
```

### Adding Dependencies

Let’s add a dependency to our application. You can find all sorts of libraries on [crates.io](https://crates.io), the package registry for Rust. In Rust, we often refer to packages as “crates.”

In this project, we’ll use a crate called `ferris-says`.

In our `Cargo.toml` file, we’ll add this information (that we got from the crate page):

```toml
[dependencies]
ferris-says = "0.3.1"
```

We can also do this by running:

```bash
cargo add ferris-says
```

Now we can run:

```bash
cargo build
```

...and Cargo will install our dependency for us.

You’ll see that running this command created a new file for us, `Cargo.lock`. This file is a log of the exact versions of the dependencies we are using locally.

To use this dependency, we can open `main.rs`, remove everything that’s in there (it’s just another example), and add this line to it:

```rust
use ferris_says::say;
```

This line means that we can now use the `say` function that the `ferris-says` crate exports for us.

### A Small Rust Application

Now let’s write a small application with our new dependency. In our `main.rs`, add the following code:

```rust
use ferris_says::say; // from the previous step
use std::io::{stdout, BufWriter};

fn main() {
    let stdout = stdout();
    let message = String::from("Hello fellow Rustaceans!");
    let width = message.chars().count();

    let mut writer = BufWriter::new(stdout.lock());
    say(&message, width, &mut writer).unwrap();
}
```

Once we save that, we can run our application by typing:

```bash
cargo run
```

Assuming everything went well, you should see your application print this to the screen:

```text
 __________________________
< Hello fellow Rustaceans! >
 --------------------------
        \
         \
            _~^~^~_
        \) /  o o  \ (/
          '_   -   _'
          / '-----' \
```
