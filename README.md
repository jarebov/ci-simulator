# Frequentist 95% Confidence Interval Visualizer

Single-page React + TypeScript app that animates repeated sampling from a finite population and draws 95% confidence intervals for the mean.

## Features

- Fixed finite population (`Npop = 10,000`) of integer sibling counts.
- Fixed true mean `mu` (red vertical line).
- Repeated random samples without replacement.
- 95% CI for each sample using Student t critical value.
- CI strip chart showing contain/miss with color coding.
- Running coverage rate display.
- Controls: sample size, repetitions, speed, Start/Pause/Step/Reset, and visualization toggles.

## How to run

1. Install dependencies:

```bash
npm install
```

2. Start dev server:

```bash
npm run dev
```

3. Open the local URL shown by Vite (usually `http://localhost:5173`).

## Build

```bash
npm run build
npm run preview
```
