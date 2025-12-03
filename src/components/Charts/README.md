# Charts Components

This directory contains data visualization components for training metrics.

## Component to Implement (TASK-008)

### MetricsChart.jsx

**Purpose**: Visualize training metrics over time (score, reward, loss, exploration)

**Props:**
```typescript
interface MetricsChartProps {
  metrics: Array<{
    episode: number;
    score: number;
    reward: number;
    loss?: number;
    epsilon?: number;    // DQN
    entropy?: number;    // PPO
  }>;
  selectedMetrics?: string[];  // Which metrics to show (default: all)
}
```

**Features to implement:**

1. **Charts** (using a charting library like Chart.js, Recharts, or Victory)
   - **Score over episodes** (line chart)
     - X-axis: Episode number
     - Y-axis: Score
     - Shows trend over training
   - **Average reward over episodes** (line chart)
     - X-axis: Episode number
     - Y-axis: Cumulative reward
   - **Loss over episodes** (line chart)
     - X-axis: Episode number
     - Y-axis: Training loss
   - **Exploration over episodes** (line chart)
     - X-axis: Episode number
     - Y-axis: Epsilon (DQN) or Entropy (PPO)

2. **Controls**
   - Toggle which metrics to display (checkboxes)
   - Zoom/pan controls (if library supports)
   - Show last N episodes slider

3. **Performance**
   - Efficiently handle large datasets (1000+ episodes)
   - Use downsampling if needed

**Data source:**
```javascript
const { metrics } = await api.getMetrics(envName, runId, 1000);
```

**Usage in App.jsx:**
```jsx
import MetricsChart from './components/Charts/MetricsChart';

// In Train tab (or new Analytics tab)
const [metrics, setMetrics] = useState([]);

useEffect(() => {
  if (runId) {
    api.getMetrics(envName, runId, 1000).then(data => setMetrics(data.metrics));
  }
}, [runId]);

<MetricsChart metrics={metrics} />
```

**Recommended libraries:**
- **Chart.js** — Popular, feature-rich, good performance
- **Recharts** — React-friendly, composable, simple API
- **Victory** — Highly customizable, React-native compatible

**Acceptance criteria:**
- Charts render correctly with sample data
- Updates when new data arrives
- Responsive design
- Smooth performance with 1000+ data points
