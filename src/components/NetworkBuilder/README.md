# NetworkBuilder Component

This component provides a visual interface for building neural network architectures.

## Component to Implement (TASK-004)

### NetworkBuilder.jsx

**Purpose**: Replace JSON-based network configuration with a visual builder

**Props:**
```typescript
interface NetworkBuilderProps {
  networkConfig: {
    layers: Array<{
      type: "dense";
      units: number;
      activation: "relu" | "leaky_relu" | "tanh" | "sigmoid" | "linear";
    }>;
  };
  onChange: (newConfig: NetworkConfig) => void;
  disabled?: boolean;
}
```

**Features to implement:**

1. **Layer List View**
   - Vertical stack of layer cards
   - Each card shows: layer index, number of units, activation function
   - Add/remove layer buttons
   - Minimum 1 layer required
   - (Optional) Drag to reorder layers

2. **Layer Editor**
   - Units: Number input (min: 1, max: 2048)
   - Activation: Dropdown with options (relu, leaky_relu, tanh, sigmoid, linear)

3. **Presets** (optional)
   - Predefined architectures: Small, Medium, Large
   - "Save as preset" button

4. **Validation**
   - Ensure at least 1 layer exists
   - Units must be positive integers

**Usage in App.jsx:**
```jsx
import NetworkBuilder from './components/NetworkBuilder/NetworkBuilder';

// In Configuration tab
<NetworkBuilder
  networkConfig={config.network_config}
  onChange={(newConfig) => setConfig({...config, network_config: newConfig})}
  disabled={isTraining}
/>
```

**Backend contract:**
The `network_config` format is defined by `backend/agents/network_builder.py` and MUST NOT be changed. Only the UI for editing it changes.
