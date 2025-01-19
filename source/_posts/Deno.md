---
title: Deno
date: 2025-01-19 21:35:10
---

## Create a React app with Vite and Deno

This tutorial will use [create-vite](https://vitejs.dev/) to quickly scaffold a Deno and React app. Vite is a build tool and development server for modern web projects. It pairs well with React and Deno, leveraging ES modules and allowing you to import React components directly.

In your terminal run the following command to create a new React app with Vite using the typescript template:

```sh
deno run -A npm:create-vite@latest --template react-ts
```

When prompted, give your app a name, and `cd` into the newly created project directory. Then run the following command to install the dependencies:

```sh
deno install
```

Now you can serve your new react app by running:

```sh
deno task dev
```

This will start the Vite server, click the output link to localhost to see your app in the browser. If you have the [Deno extension for VSCode](https://docs.deno.com/runtime/getting_started/setup_your_environment/#visual-studio-code) installed, you may notice that the editor highlights some errors in the code. This is because the app created by Vite is designed with Node in mind and so uses conventions that Deno does not (such as 'sloppy imports' - importing modules without the file extension). Disable the Deno extension for this project to avoid these errors or try out the [tutorial to build a React app with a deno.json file](https://docs.deno.com/runtime/tutorials/how_to_with_npm/react/).