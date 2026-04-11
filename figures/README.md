# README figures

PNGs here are **web-resolution** copies (long edge 1280px) so `git push` stays small. Full-resolution sources stay under `paper/figures/` for the PDF.

After you regenerate paper figures, refresh and downsample:

```bash
cp paper/figures/demo_screenshot.png paper/figures/qual_*.png paper/figures/skm_filmstrip.png figures/
for f in figures/demo_screenshot.png figures/qual_instant.png figures/qual_recent.png figures/qual_historical.png; do
  sips -Z 1280 "$f" --out "$f"
done
```
