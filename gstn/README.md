# GramStatTexNet — project page

Static project page for *GramStatTexNet: Efficient, Interpretable, and Neuro-Inspired Texture Analysis-by-Synthesis* (NeurIPS 2026).
Served at **https://vdutell.github.io/gstn/**. Plain HTML, no build step.

## Publish (one time)

GitHub serves any repo with Pages enabled at `vdutell.github.io/<repo-name>`, so this repo just needs to be called `gstn`.

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
index.html                     the page (all CSS/JS inline)
static/gstn_neurips2026.pdf    paper (images downsampled: 13 MB instead of 61 MB)
static/images/fig*.jpg         figures pulled from the paper
static/videos/                 put the dynamic-texture videos here
.nojekyll                      tells GitHub Pages to serve files as-is
```

## To do

- **Videos.** Add `static/videos/<name>_target.mp4` and `static/videos/<name>_gstn.mp4` for
  `grass`, `fire`, `fence`, `blooms`, `tree`. Missing files show a grey placeholder, so you can add them one at a time.
  Keep them small, H.264 and silent so they autoplay everywhere:
  ```bash
  ffmpeg -i in.avi -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -an \
         -movflags +faststart -vf "scale=256:-2" grass_target.mp4
  ```
- **arXiv / Code buttons.** Search `TODO` in `index.html`; replace `href="#"` with the real link and remove `class="disabled"` (and the `<small>soon</small>`).
- **Author links.** Only Vasha, Bill Freeman and Ruth Rosenholtz are linked; wrap other names in `<a href="…">` as needed.
