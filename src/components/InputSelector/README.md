# InputSelector Component

This component provides an interface for configuring what the agent observes (input/observation space).

## Component to Implement (TASK-006)

### InputSelector.jsx

**Purpose**: Allow users to configure vision mode and other observation settings

**Props:**
```typescript
interface InputSelectorProps {
  envConfig: {
    vision: number;     // 0=immediate danger, >0=window size, -1=full grid
    seg_size: number;   // Number of body segments to track
  };
  onChange: (newConfig: EnvConfig) => void;
  disabled?: boolean;
}
```

**Features to implement:**

1. **Vision Mode Selector**
   - Radio buttons or dropdown with options:
     - **Immediate Danger** (`vision: 0`) — 4 directions (up, down, left, right)
     - **3×3 Window** (`vision: 1`) — 8 cells around head
     - **5×5 Window** (`vision: 2`) — 24 cells around head
     - **Full Grid** (`vision: -1`) — Entire grid visible
   - Show description for each mode
   - Show resulting observation space size

2. **Body Segment Tracking**
   - Slider or number input (range: 1-10)
   - Explanation: "Number of body segments tracked in observation (helps infer body shape)"

3. **Observation Space Preview**
   - Show calculated total observation size
   - Break down by components:
     - Food position: 2
     - Direction: 2
     - Hunger: 1
     - Body segments: `seg_size * 2`
     - Danger/vision: varies by mode
   - Example: "Total observation size: 31 features"

**Vision modes explained:**
- **Immediate Danger** (vision=0): Only checks 4 adjacent cells for walls/body (4 features)
- **3×3 Window** (vision=1): Sees cells in 3×3 grid around head (8 features, excluding head cell)
- **5×5 Window** (vision=2): Sees cells in 5×5 grid around head (24 features)
- **Full Grid** (vision=-1): Sees entire grid (grid_size² features)

**Usage in App.jsx:**
```jsx
import InputSelector from './components/InputSelector/InputSelector';

// In Configuration tab
<InputSelector
  envConfig={config.env_config}
  onChange={(newConfig) => setConfig({...config, env_config: {...config.env_config, ...newConfig}})}
  disabled={isTraining}
/>
```

**Backend contract:**
The `vision` and `seg_size` fields are already supported by `SnakeEnv`. No backend changes required.
