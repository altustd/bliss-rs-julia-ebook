# BLISS-2026 Interactive Visualizations

This directory contains interactive HTML visualizations for the BLISS-2026 thesis.

## Files

### `dashboard.html`
Interactive data visualization dashboard showing:
- GP surrogate surfaces with uncertainty bands
- Convergence comparison (BLISS-RS vs BLISS-2026)
- Expected Improvement acquisition functions
- Adaptive vs fixed sampling strategies
- Surrogate approximation error comparison
- Bi-level convergence (system vs local optimization)

**Usage:** Open directly in a browser, or host on a web server.

### `presentation.html`
15-slide presentation deck summarizing BLISS-2026:
- Problem motivation and MDO challenges
- Bi-level decomposition architecture
- Evolution from 2002 to 2026
- GP surrogate theory and advantages
- Supersonic business jet test case results
- Interactive embedded visualizations

**Controls:**
- Arrow keys or click to navigate
- Print to PDF: Cmd/Ctrl + P
- Fullscreen: F key

**Requirements:** Needs `deck-stage.js` in the same directory.

### `deck-stage.js`
Slide deck framework component. Required by `presentation.html`.

## Deployment

### GitHub Pages
1. Push this directory to your repo
2. Enable GitHub Pages in Settings → Pages
3. Set source to main branch `/visualizations` or `/docs`
4. Access at: `https://yourusername.github.io/reponame/visualizations/presentation.html`

### Local Server
```bash
cd visualizations
python -m http.server 8000
# Open http://localhost:8000/presentation.html
```

### Static Hosting
Upload all three files to any static host (Netlify, Vercel, troyaltus.com, etc.)

## Browser Compatibility

Modern browsers only (Chrome, Firefox, Safari, Edge). Requires:
- ES6+ JavaScript
- Canvas API
- CSS Grid

## License

Part of the BLISS-2026 project. See parent repository for license details.
