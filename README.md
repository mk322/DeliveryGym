# DeliveryGym project website

An English academic project website with a dedicated leaderboard page. Plain HTML, CSS, and JavaScript; no build step, server-side code, fonts, chart libraries, or external runtime dependencies. All local assets use relative URLs, including when hosted at a GitHub Pages subpath.

## Open locally

Open `index.html` directly for a basic preview. To use an HTTP origin, run from the repository root:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Then visit `http://127.0.0.1:8765/website/`. Stop with Ctrl+C. Copying BibTeX requires clipboard access; if unavailable, the page selects the citation and explains how to copy manually. The public project page is `https://mk322.github.io/DeliveryGym/`.

## Files and ownership

| File | Purpose |
| --- | --- |
| `index.html` | Page structure, authors, affiliations, overview text, resource links |
| `styles.css` | Responsive layout, chart styles, research color palette, reduced-motion rules |
| `leaderboard.html`, `leaderboard.js` | Dedicated Waypoint/Any-Point leaderboard and trained-policy comparison |
| `results-data.js` | Authoritative numerical dataset; source labels, splits, units, and comparison conditions |
| `app.js` | Accessible figure dialog and citation copy |
| `assets/` | Launch-film MP4 and poster, hero/social artwork, exact source overview, rendered active paper figures, original figure PDFs, favicon |
| `tests/check-static.cjs` | Dependency-free source/data consistency check; does not launch a browser |
| `docs/PLAN.md` | Current implementation decisions and remaining verification |
| `docs/HISTORY.md` | Concise iteration history |

## Maintain results

1. Read the current active content of `../paper/main.tex`, excluding `%` comments and `\iffalse` branches. The eight-author commented block is the explicitly requested exception for author attribution.
2. Update `results-data.js`; it drives the explorer and benchmark table. Benchmark tuples are `[income, delivered, onTime, redLight, obstacle]`. Null means unreported, never zero. The selected-policy test table has the historical label `tab:final-test-estimates`, but is active and reports the measured values used by the current paper.
3. Keep reward experiments, curriculum experiments, training probes, and validation scaling separate. Never multiply/add the two headline percentages, import validation service/safety values into test rows, or invent Any-Point RL results.
4. Update source metadata and comparison text if the protocol changes. Do not infer training curve points from plotted lines. Display original curves with their validation label.
5. Run `node tests/check-static.cjs` from `website`, then perform the manual checks in `docs/PLAN.md` when requested.

The diagnostics explorer preserves the integer percentage labels printed in the active original figure (e.g. 76%), rather than replacing them with recalculated decimals. Relative gains elsewhere are calculated from the reported means. The scaling chart uses equally spaced discrete conditions and explicitly labels this; it does not imply linear task-pool spacing or interpolate unreported data.

## Demo film

`assets/deliverygym-demo.mp4` is the 106-second, 1080p demo used by the native HTML video player. The opening asks whether RL can teach an embodied agent to deliver food in Paris, shows orders and six constraints, illustrates a humanoid courier trajectory with traffic/closure counterfactuals, then explains profit-based RL and the measured improvement. Adaptive environment, scaling and closing follow. The street reconstruction uses generated photographic background plates and animated 3D actors; it is not recorded UE gameplay. See `../demo/remotion/PRODUCTION-v6.md` for asset provenance and reproduction. `assets/video-poster.jpg` is its poster.

## Refresh figures

`assets/overview.png` is a byte-identical copy of `paper/figures/figure1_overview.png`. All other figure PDFs are copied from the correspondingly named active paper files; PNGs are their web renditions. Keep the PDF links for full-resolution inspection. With Poppler available, regenerate a rendition, for example:

```powershell
pdftoppm -png -singlefile -scale-to 2200 assets/figure2_framework.pdf assets/figure2_framework
```

Update image dimensions if the source aspect ratio changes. Do not substitute the unused `Figure2_framework.png`, draft figures, estimated figures, archived text, or video-source assets. Author homepage links require evidence of identity; only Lianhui Qin currently has a link verified through the DeliveryBench reference website.

## Validation status

Static data/source/link checks and the video export checks pass. The project is published from the `gh-pages` branch of `mk322/DeliveryGym` at `https://mk322.github.io/DeliveryGym/`.
