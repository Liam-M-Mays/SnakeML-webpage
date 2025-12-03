# RewardDesigner Component

This component provides an interface for configuring rewards, including event-based formulas.

## Component to Implement (TASK-005)

### RewardDesigner.jsx

**Purpose**: Replace simple numeric reward inputs with an advanced event-based reward designer

**Props:**
```typescript
interface RewardDesignerProps {
  rewardConfig: {
    // Legacy: simple numeric rewards
    apple: number;
    death_wall: number;
    death_self: number;
    death_starv: number;
    step: number;

    // New: event-based formulas (optional)
    formulas?: Array<{
      event: string;
      formula: string;
      enabled: boolean;
    }>;
  };
  onChange: (newConfig: RewardConfig) => void;
  disabled?: boolean;
}
```

**Features to implement:**

1. **Simple Mode** (default)
   - Numeric inputs for each event type
   - Same as current UI in App.jsx

2. **Advanced Mode** (toggle)
   - Event-based reward editor
   - For each event:
     - Enable/disable toggle
     - Formula input field
     - Available variables dropdown/suggestions
     - Preview with example values

3. **Available Events**
   - `apple_eaten` — When snake eats food
   - `death_wall` — When snake hits wall
   - `death_self` — When snake hits itself
   - `death_starv` — When snake starves
   - `step` — Per step (every action)

4. **Available Variables** (for formulas)
   - `snake_length` — Current snake length
   - `hunger` — Steps since last food
   - `max_hunger` — Starvation limit
   - `grid_size` — Grid dimensions
   - `score` — Current score
   - `distance_to_food` — Manhattan distance to food (TODO in backend)

5. **Formula Examples**
   - `1.0 + 0.1 * snake_length` — Reward increases with snake length
   - `-0.001 * (1 + hunger / max_hunger)` — Step penalty increases with hunger

**Usage in App.jsx:**
```jsx
import RewardDesigner from './components/RewardDesigner/RewardDesigner';

// In Configuration tab
<RewardDesigner
  rewardConfig={config.reward_config}
  onChange={(newConfig) => setConfig({...config, reward_config: newConfig})}
  disabled={isTraining}
/>
```

**Backend changes required:**
See TASK-005 in `docs/AGENT_TASKS.md` for backend implementation details. The backend must support evaluating formulas safely.

**Security note:**
Formulas are evaluated on the backend. Ensure safe evaluation (no `eval()` calls with arbitrary user code).
