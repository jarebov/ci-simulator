import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type Speed = 'slow' | 'medium' | 'fast';
type ConfidenceLevel = 0.99 | 0.95 | 0.9;

type IntervalRecord = {
  lower: number;
  upper: number;
  mean: number;
  containsMu: boolean;
};

type DotPoint = {
  x: number;
  jitterX: number;
  jitterY: number;
};

const NPOP = 10_000;
const MAX_POP_DOTS = 2_000;
const DOT_JITTER_X_HALF_RANGE = 0.18;
const TOP_DOT_MAX_RADIUS_PX = 3.3;
const TOP_PLOT_RIGHT_PAD_PX = Math.ceil(TOP_DOT_MAX_RADIUS_PX + 2);
const SPEED_MS: Record<Speed, number> = {
  slow: 900,
  medium: 350,
  fast: 120,
};

const siblingDistribution: Array<{ value: number; p: number }> = [
  { value: 0, p: 0.21 },
  { value: 1, p: 0.39 },
  { value: 2, p: 0.25 },
  { value: 3, p: 0.1 },
  { value: 4, p: 0.035 },
  { value: 5, p: 0.015 },
];

function generatePopulation(size: number): number[] {
  const generateOnce = () => {
    const pop: number[] = new Array(size);
    for (let i = 0; i < size; i += 1) {
      const r = Math.random();
      let picked = cumulative[cumulative.length - 1].value;
      for (const c of cumulative) {
        if (r <= c.cdf) {
          picked = c.value;
          break;
        }
      }
      pop[i] = picked;
    }
    return pop;
  };

  const cumulative: Array<{ value: number; cdf: number }> = [];
  let running = 0;
  for (const d of siblingDistribution) {
    running += d.p;
    cumulative.push({ value: d.value, cdf: running });
  }

  const targetMu = 1.4;
  const tolerance = 0.02;
  let best = generateOnce();
  let bestGap = Math.abs(mean(best) - targetMu);

  for (let attempt = 0; attempt < 24; attempt += 1) {
    if (bestGap <= tolerance) break;
    const candidate = generateOnce();
    const gap = Math.abs(mean(candidate) - targetMu);
    if (gap < bestGap) {
      best = candidate;
      bestGap = gap;
    }
  }

  return best;
}

function sampleIndicesWithoutReplacement(populationSize: number, n: number): number[] {
  const picked = new Set<number>();
  while (picked.size < n) {
    picked.add(Math.floor(Math.random() * populationSize));
  }
  return Array.from(picked);
}

function mean(values: number[]): number {
  let total = 0;
  for (const v of values) total += v;
  return total / values.length;
}

function sampleStd(values: number[], xbar: number): number {
  if (values.length < 2) return 0;
  let ss = 0;
  for (const v of values) {
    const d = v - xbar;
    ss += d * d;
  }
  return Math.sqrt(ss / (values.length - 1));
}

// Acklam approximation for inverse normal CDF.
function inverseNormalCdf(p: number): number {
  if (p <= 0 || p >= 1) throw new Error('p must be in (0,1)');

  const a = [
    -3.969683028665376e1,
    2.209460984245205e2,
    -2.759285104469687e2,
    1.38357751867269e2,
    -3.066479806614716e1,
    2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1,
    1.615858368580409e2,
    -1.556989798598866e2,
    6.680131188771972e1,
    -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3,
    -3.223964580411365e-1,
    -2.400758277161838,
    -2.549732539343734,
    4.374664141464968,
    2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3,
    3.224671290700398e-1,
    2.445134137142996,
    3.754408661907416,
  ];

  const plow = 0.02425;
  const phigh = 1 - plow;

  if (p < plow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  if (p > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }

  const q = p - 0.5;
  const r = q * q;
  return (
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
  );
}

function tCriticalForLevel(df: number, level: number): number {
  const alpha = 1 - level;
  const p = 1 - alpha / 2;
  const z = inverseNormalCdf(p);
  if (df <= 0) return z;
  const z2 = z * z;
  const z3 = z2 * z;
  const z5 = z3 * z2;
  const z7 = z5 * z2;
  const g1 = (z3 + z) / (4 * df);
  const g2 = (5 * z5 + 16 * z3 + 3 * z) / (96 * df * df);
  const g3 = (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * df * df * df);
  return z + g1 + g2 + g3;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function stableUnitNoise(seed: number): number {
  const x = Math.sin(seed * 12.9898) * 43758.5453123;
  return x - Math.floor(x);
}

function setupHiDpiCanvas(canvas: HTMLCanvasElement) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  const fallbackWidth = Number(canvas.getAttribute('width')) || 1;
  const fallbackHeight = Number(canvas.getAttribute('height')) || 1;
  const cssWidth = Math.max(1, canvas.clientWidth || fallbackWidth);
  const cssHeight = Math.max(1, canvas.clientHeight || fallbackHeight);
  const dpr = window.devicePixelRatio || 1;
  const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
  const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));

  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: cssWidth, height: cssHeight };
}

export default function App() {
  const [population] = useState<number[]>(() => generatePopulation(NPOP));
  const [sampleSize, setSampleSize] = useState(100);
  const [targetRepetitions, setTargetRepetitions] = useState(100);
  const [speed, setSpeed] = useState<Speed>('medium');
  const [confidenceLevel, setConfidenceLevel] = useState<ConfidenceLevel>(0.95);
  const [showPopulationDots, setShowPopulationDots] = useState(true);
  const [showCurrentSample, setShowCurrentSample] = useState(true);
  const [freezeAfterOne, setFreezeAfterOne] = useState(false);

  const [isRunning, setIsRunning] = useState(false);
  const [intervals, setIntervals] = useState<IntervalRecord[]>([]);
  const [containsCount, setContainsCount] = useState(0);
  const [currentSample, setCurrentSample] = useState<DotPoint[]>([]);
  const [currentInterval, setCurrentInterval] = useState<IntervalRecord | null>(null);

  const topCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const bottomCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const mu = useMemo(() => mean(population), [population]);
  const xMin = useMemo(() => Math.min(...population), [population]);
  const xMax = useMemo(() => Math.max(...population), [population]);

  const populationDots = useMemo(() => {
    const stride = Math.max(1, Math.floor(population.length / MAX_POP_DOTS));
    const points: DotPoint[] = [];
    for (let i = 0; i < population.length; i += stride) {
      points.push({
        x: population[i],
        jitterX: (stableUnitNoise(i + 11) - 0.5) * 0.36,
        jitterY: stableUnitNoise(i + 97),
      });
    }
    return points;
  }, [population]);

  useEffect(() => {
    if (intervals.length > targetRepetitions) {
      const trimmed = intervals.slice(0, targetRepetitions);
      const c = trimmed.reduce((acc, v) => acc + (v.containsMu ? 1 : 0), 0);
      setIntervals(trimmed);
      setContainsCount(c);
      if (trimmed.length >= targetRepetitions) setIsRunning(false);
    }
  }, [intervals, targetRepetitions]);

  const runOneRepetition = useCallback(() => {
    if (intervals.length >= targetRepetitions) {
      setIsRunning(false);
      return;
    }

    const sampledIndices = sampleIndicesWithoutReplacement(population.length, sampleSize);
    const sample = sampledIndices.map((idx) => population[idx]);
    const xbar = mean(sample);
    const s = sampleStd(sample, xbar);
    const se = s / Math.sqrt(sampleSize);
    const tcrit = tCriticalForLevel(sampleSize - 1, confidenceLevel);
    const margin = tcrit * se;
    const lower = xbar - margin;
    const upper = xbar + margin;
    const containsMu = lower <= mu && mu <= upper;

    const newInterval: IntervalRecord = { lower, upper, mean: xbar, containsMu };

    const sampledPoints = sampledIndices.map((idx) => ({
      x: population[idx],
      jitterX: (stableUnitNoise(idx + 11) - 0.5) * 0.36,
      jitterY: stableUnitNoise(idx + 97),
    }));

    setCurrentSample(sampledPoints);
    setCurrentInterval(newInterval);
    setIntervals((prev) => [...prev, newInterval]);
    setContainsCount((prev) => prev + (containsMu ? 1 : 0));

    if (freezeAfterOne) {
      setIsRunning(false);
    }
  }, [confidenceLevel, freezeAfterOne, intervals.length, mu, population, sampleSize, targetRepetitions]);

  useEffect(() => {
    setIsRunning(false);
    setIntervals([]);
    setContainsCount(0);
    setCurrentSample([]);
    setCurrentInterval(null);
  }, [confidenceLevel]);

  useEffect(() => {
    if (!isRunning) return;
    const id = window.setInterval(() => {
      runOneRepetition();
    }, SPEED_MS[speed]);
    return () => window.clearInterval(id);
  }, [isRunning, runOneRepetition, speed]);

  useEffect(() => {
    const canvas = topCanvasRef.current;
    if (!canvas) return;
    const setup = setupHiDpiCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const margin = { left: 54, right: 20, top: 20, bottom: 38 };
    const axisY = height - margin.bottom;
    const plotLeft = margin.left;
    const plotRight = width - margin.right - TOP_PLOT_RIGHT_PAD_PX;
    const domainMin = xMin;
    const domainMax = xMax + DOT_JITTER_X_HALF_RANGE;

    const mapX = (v: number) => {
      const frac = (v - domainMin) / (domainMax - domainMin || 1);
      return plotLeft + frac * (plotRight - plotLeft);
    };

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#d0d7de';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotLeft, axisY);
    ctx.lineTo(plotRight, axisY);
    ctx.stroke();

    ctx.fillStyle = '#4b5563';
    ctx.font = '12px sans-serif';
    for (let t = Math.floor(xMin); t <= Math.ceil(xMax); t += 1) {
      const tx = mapX(t);
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath();
      ctx.moveTo(tx, axisY);
      ctx.lineTo(tx, axisY + 5);
      ctx.stroke();
      ctx.fillText(String(t), tx - 4, axisY + 18);
    }

    if (showPopulationDots) {
      ctx.fillStyle = 'rgba(107,114,128,0.42)';
      for (const p of populationDots) {
        const x = mapX(p.x + p.jitterX);
        const y = axisY - 8 - p.jitterY * 90;
        ctx.beginPath();
        ctx.arc(x, y, 2.1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (showCurrentSample && currentSample.length > 0) {
      for (const p of currentSample) {
        const x = mapX(p.x + p.jitterX);
        const y = axisY - 8 - p.jitterY * 90;
        ctx.strokeStyle = 'rgba(239,246,255,0.95)';
        ctx.lineWidth = 1.1;
        ctx.beginPath();
        ctx.arc(x, y, 3.3, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = 'rgba(37,99,235,0.92)';
        ctx.beginPath();
        ctx.arc(x, y, 2.7, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (currentInterval) {
      const y = margin.top + 24;
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(mapX(currentInterval.lower), y);
      ctx.lineTo(mapX(currentInterval.upper), y);
      ctx.stroke();

      ctx.strokeStyle = '#1d4ed8';
      ctx.lineWidth = 2;
      const meanX = mapX(currentInterval.mean);
      ctx.beginPath();
      ctx.moveTo(meanX, y - 14);
      ctx.lineTo(meanX, y + 14);
      ctx.stroke();
    }

    const muX = mapX(mu);
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(muX, margin.top);
    ctx.lineTo(muX, axisY + 3);
    ctx.stroke();

    ctx.fillStyle = '#dc2626';
    ctx.font = 'bold 12px sans-serif';
    ctx.fillText('\u03bc (true mean)', clamp(muX - 42, plotLeft, plotRight - 85), margin.top + 12);
  }, [currentInterval, currentSample, mu, populationDots, showCurrentSample, showPopulationDots, xMax, xMin]);

  useEffect(() => {
    const canvas = bottomCanvasRef.current;
    if (!canvas) return;
    const setup = setupHiDpiCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const margin = { left: 54, right: 20, top: 16, bottom: 34 };
    const drawableH = height - margin.top - margin.bottom;
    const rowH = drawableH / Math.max(1, targetRepetitions);

    const mapX = (v: number) => {
      const frac = (v - xMin) / (xMax - xMin || 1);
      return margin.left + frac * (width - margin.left - margin.right);
    };

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, width, height);

    const muX = mapX(mu);
    ctx.strokeStyle = '#dc2626';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(muX, margin.top);
    ctx.lineTo(muX, height - margin.bottom);
    ctx.stroke();

    for (let i = 0; i < intervals.length; i += 1) {
      const iv = intervals[i];
      const y = margin.top + (i + 0.5) * rowH;
      ctx.strokeStyle = iv.containsMu ? '#16a34a' : '#ef4444';
      ctx.lineWidth = rowH > 3 ? 2 : 1;
      ctx.beginPath();
      ctx.moveTo(mapX(iv.lower), y);
      ctx.lineTo(mapX(iv.upper), y);
      ctx.stroke();
    }

    ctx.strokeStyle = '#d0d7de';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();

    ctx.fillStyle = '#4b5563';
    ctx.font = '12px sans-serif';
    for (let t = Math.floor(xMin); t <= Math.ceil(xMax); t += 1) {
      const tx = mapX(t);
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath();
      ctx.moveTo(tx, height - margin.bottom);
      ctx.lineTo(tx, height - margin.bottom + 5);
      ctx.stroke();
      ctx.fillText(String(t), tx - 4, height - margin.bottom + 18);
    }

    ctx.fillStyle = '#6b7280';
    ctx.font = '11px sans-serif';
    ctx.fillText('Repetition index', 6, margin.top + 10);
  }, [intervals, mu, targetRepetitions, xMax, xMin]);

  const coverage = intervals.length === 0 ? 0 : containsCount / intervals.length;

  const handleStart = () => {
    if (intervals.length >= targetRepetitions) return;
    setIsRunning(true);
  };

  const handlePause = () => setIsRunning(false);

  const handleStep = () => {
    if (intervals.length >= targetRepetitions) return;
    setIsRunning(false);
    runOneRepetition();
  };

  const handleReset = () => {
    setIsRunning(false);
    setIntervals([]);
    setContainsCount(0);
    setCurrentSample([]);
    setCurrentInterval(null);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Confidence Interval Simulator</h1>
          <p className="subtitle">Econ 1117 – Yale University</p>
        </div>
        <img className="yale-logo" src={import.meta.env.BASE_URL + 'yale_logo.png'} alt="Yale University logo" />
      </header>

      <div className="description">
       <p>
          This simulator illustrates how to interpret confidence intervals.
        </p>

        <p>
          Consider the population of Yale students. Let X denote the number of siblings of a randomly selected student. 
          The population mean is μ = 1.4 (that is, E[X] = μ = 1.4). 
          This value is fixed and does not change.
        </p>

        <p>
          The first panel below shows the population distribution of X, represented by the gray circles. 
          The red vertical line marks the true mean μ.
        </p>

        <p>
          The simulation repeatedly draws random samples from this population. 
          In each repetition, a sample (blue dots) is drawn, the sample mean is computed, and a confidence interval is constructed. 
          Because the sample changes from one repetition to the next, the interval changes as well. 
          Over many repetitions, a (1−α)% confidence procedure captures the true mean approximately (1−α)% of the time.
        </p>
      </div>

      <div className="controls">
        <label>
          Sample size n: <strong>{sampleSize}</strong>
          <input
            type="range"
            min={10}
            max={500}
            step={1}
            value={sampleSize}
            onChange={(e) => setSampleSize(Number(e.target.value))}
          />
        </label>

        <label>
          Repetitions: <strong>{targetRepetitions}</strong>
          <input
            type="range"
            min={10}
            max={500}
            step={1}
            value={targetRepetitions}
            onChange={(e) => setTargetRepetitions(Number(e.target.value))}
          />
        </label>

        <label>
          Speed
          <select value={speed} onChange={(e) => setSpeed(e.target.value as Speed)}>
            <option value="slow">Slow</option>
            <option value="medium">Medium</option>
            <option value="fast">Fast</option>
          </select>
        </label>

        <label>
          Confidence level
          <select
            value={String(confidenceLevel)}
            onChange={(e) => setConfidenceLevel(Number(e.target.value) as ConfidenceLevel)}
          >
            <option value="0.99">99%</option>
            <option value="0.95">95%</option>
            <option value="0.9">90%</option>
          </select>
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={showPopulationDots}
            onChange={(e) => setShowPopulationDots(e.target.checked)}
          />
          Show population dots
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={showCurrentSample}
            onChange={(e) => setShowCurrentSample(e.target.checked)}
          />
          Show current sample
        </label>

        <label className="check">
          <input
            type="checkbox"
            checked={freezeAfterOne}
            onChange={(e) => setFreezeAfterOne(e.target.checked)}
          />
          Freeze after one interval
        </label>

        <div className="buttons">
          <button onClick={handleStart} disabled={isRunning || intervals.length >= targetRepetitions}>
            Start
          </button>
          <button onClick={handlePause} disabled={!isRunning}>
            Pause
          </button>
          <button onClick={handleStep} disabled={intervals.length >= targetRepetitions}>
            Step
          </button>
          <button onClick={handleReset}>Reset</button>
        </div>
      </div>

      <div className="stats">
        <div>Population size: {NPOP.toLocaleString()}</div>
        <div>True mean μ: 1.4</div>
        <div>Intervals drawn: {intervals.length}</div>
        <div>
          Coverage so far: {(coverage * 100).toFixed(1)}% ({containsCount}/{intervals.length || 0})
        </div>
      </div>

      <section className="panel">
        <h2>Population Axis and Current Draw</h2>
        <div className="chart-wrap chart-wrap-top">
          <canvas ref={topCanvasRef} width={980} height={220} />
        </div>
      </section>

      <section className="panel">
        <h2>Repeated {(confidenceLevel * 100).toFixed(0)}% Confidence Intervals</h2>
        <div className="chart-wrap chart-wrap-bottom">
          <canvas ref={bottomCanvasRef} width={980} height={460} />
        </div>
      </section>

      <footer className="footer-credit">
        Interactive CI visualization by{' '}
        <a
          href="https://www.jarellanobover.com/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Open Jaime Arellano-Bover’s website in a new tab"
        >
          Jaime Arellano-Bover
        </a>
      </footer>
    </div>
  );
}
