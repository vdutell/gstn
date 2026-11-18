# GramStatTexNet — project page

Static project page for *GramStatTexNet: Efficient, Interpretable, and Neuro-Inspired Texture Analysis-by-Synthesis* (NeurIPS 2026).
Served at **https://vdutell.github.io/gstn/**. Plain HTML, no build step.

## Publish

This repo now holds **both** the project page and the codebase. The page is served from
the `docs/` folder, not the repo root, which leaves the root free for the code and keeps
the code files from being web-served. The published URL is unchanged, because a Pages URL
depends on the repo name, not on which folder the site lives in.

    Settings -> Pages -> Deploy from a branch -> main / docs

Originally published as a page-only repo like this:

```bash
cd gstn
git init -b main
git add .
git commit -m "GSTN project page"

# with the GitHub CLI:
gh repo create vdutell/gstn --public --source . --push
gh api -X POST repos/vdutell/gstn/pages -f "source[branch]=main" -f "source[path]=/"

# or without it: create an empty public repo named "gstn" on github.com, then
#   git remote add origin git@github.com:vdutell/gstn.git && git push -u origin main
# and turn on Settings → Pages → Deploy from a branch → main / (root)
```

The page goes live a minute or two later.

## Layout

```
docs/index.html                     the page (all CSS/JS inline)
docs/static/gstn_neurips2026.pdf    paper (images downsampled: 13 MB instead of 61 MB)
docs/static/images/fig*.jpg         figures pulled from the paper
docs/static/videos/*.mp4            the ten dynamic-texture tiles
docs/.nojekyll                      tells GitHub Pages to serve files as-is

Everything outside docs/ is the codebase; see the repository README.
```

## To do

- ~~**Videos.**~~ Done. All ten tiles are in `static/videos/`, remuxed from the NeurIPS
  supplement package (`gstn_dynca_video_comparison/individual/`) with `-c:v copy -an
  -movflags +faststart`, so they carry no extra compression generation. 256x256, silent,
  1.33 s; the page's JS loops them.
- ~~**Code button.**~~ Done, points at this repo, which now holds the codebase as well as
  the page.
- **arXiv / OpenReview button.** Still `href="#"` with `class="disabled"`. Search `TODO` in
  `index.html`.
- **Author links.** Only Vasha, Bill Freeman and Ruth Rosenholtz are linked; wrap other names in `<a href="…">` as needed.
