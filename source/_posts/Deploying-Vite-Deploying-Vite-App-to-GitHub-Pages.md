---
title: Deploying Vite Deploying Vite App to GitHub Pages
tags:
  - Vite
category:
  - Vite
date: 2025-01-21 11:00:21
---


[Reference](https://medium.com/@aishwaryaparab1/deploying-vite-deploying-vite-app-to-github-pages-166fff40ffd3) 

# Step 1: Initialize Git Repository

Run the following commands to initialize a git repository in your Vite app and push your existing code to a remote repository on GitHub.

```bash
git init
git add .
git commit -m "initial-commit"
git branch -M main
git remote add origin http://github.com/{username}/{repo-name}.git
git push -u origin main
```

# Step 2: Update vite.config.js

Add the base URL in this file by setting the **base** as **“/{repo-name}/”**. For example, if your repository’s name is **book-landing-page** then set the **base** like this:

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/book-landing-page/"
})
```

# Step 3: Install gh-pages

Install **gh-pages** package as a dev dependency.

```bash
npm install gh-pages --save-dev
```

# Step 4: Update package.json

Update package.json with the following **predeploy** and **deploy** scripts.

```json
"scripts": {
    "predeploy" : "npm run build",
    "deploy" : "gh-pages -d dist",
    ...
}
```

Add the complete website URL by setting **homepage** in package.json

```
"homepage": "https://{username}.github.io/{repo-name}/"
```

Thus, your updated package.json will look like this:

```json
{
  "name": "book-product",
  "private": true,
  "version": "0.0.0",
  "homepage": "https://aishwaryaparab.github.io/book-landing-page/",
  "type": "module",
  "scripts": {
    "predeploy" : "npm run build",
    "deploy" : "gh-pages -d dist",
    "dev": "vite",
    "build": "vite build",
    ...
}
```

# Step 5: Run Deploy

If you’ve made it till here, you’re almost there. Run the final command:

```bash
npm run deploy
```

And you’re done!

One last step though!

Navigate to your remote repository on GitHub -> Settings -> Pages (left sidebar). Select source as “Deploy from branch” and branch as “gh-pages”.


![GitHub Pages Settings](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ybobpluBHeEmanhK7DmdNw.png)

GitHub Pages also allows you to set up your own custom domain. 🙌

Have some patience, wait for a few minutes and soon, your site will be live at https://{username}.github.io/{repo-name}/