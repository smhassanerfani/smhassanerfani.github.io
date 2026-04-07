# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Personal academic website for Mohammad H. Erfani (Machine Learning Scientist at CW3E, Scripps/UCSD). Static site hosted on GitHub Pages at `smhassanerfani.github.io`. No build system, no package manager — just HTML/CSS/JS served directly.

## Development

Open `index.html` in a browser. No build step, no dev server required. Jekyll is disabled (`.nojekyll`).

## Architecture

- **Single stylesheet:** `css/style.css` — all pages share this. Max-width layout at 1000px, primary blue `#0056b3`, gold accent `#f7b731`.
- **Single script:** `js/modal.js` — fetches conference abstract HTML into a modal overlay. Only used on `pages/conferences.html`.
- **CDN dependencies:** Font Awesome 7.0.1 (icons), MathJax 3 (math in blog posts only).

## Page Template Convention

Every page follows this structure:
1. Shared `<header><nav>` with links: Home, Publications, Conferences, Projects, Blog
2. `<main>` with page content
3. CSS linked via relative path to `css/style.css` (e.g., `../css/style.css` from pages/)

When adding a new page under `pages/`, use relative paths (`../css/style.css`, `../js/modal.js`, `../index.html`). For sub-pages (e.g., `pages/blog/`), go two levels up (`../../css/style.css`).

## Content Sections

- `pages/publications.html` — publication list with DOI links
- `pages/conferences.html` — conference abstracts with modal viewer
- `pages/projects.html` → `pages/projects/*.html` — detailed project pages
- `pages/blog.html` → `pages/blog/*.html` — blog posts (may include MathJax)
- `pages/blog/code/` — Python code examples referenced by blog posts

## SEO

`sitemap.xml` and `robots.txt` are maintained manually. Update `sitemap.xml` when adding new pages. Google Search Console verification via `google3aa92baef8f2baed.html`.
