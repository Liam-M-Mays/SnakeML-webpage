/**
 * Validation utilities for form inputs and parameters.
 */

/**
 * Validates an agent/model name.
 * @param {string} name - The name to validate
 * @returns {{ valid: boolean, error: string | null }}
 */
export function validateAgentName(name) {
  if (!name || typeof name !== 'string') {
    return { valid: false, error: 'Name is required' }
  }

  const trimmed = name.trim()

  if (trimmed.length === 0) {
    return { valid: false, error: 'Name cannot be empty' }
  }

  if (trimmed.length > 50) {
    return { valid: false, error: 'Name must be 50 characters or less' }
  }

  // Allow alphanumeric, spaces, hyphens, underscores
  if (!/^[a-zA-Z0-9\s\-_]+$/.test(trimmed)) {
    return { valid: false, error: 'Name can only contain letters, numbers, spaces, hyphens, and underscores' }
  }

  return { valid: true, error: null }
}

/**
 * Validates a hyperparameter value.
 * @param {number} value - The value to validate
 * @param {object} config - The parameter config with min, max, step
 * @returns {{ valid: boolean, error: string | null, correctedValue: number }}
 */
export function validateHyperparameter(value, config) {
  const { min, max, step, label } = config

  // Check if it's a valid number
  if (typeof value !== 'number' || isNaN(value)) {
    return { valid: false, error: `${label} must be a number`, correctedValue: config.value }
  }

  // Check min bound
  if (min !== undefined && value < min) {
    return { valid: false, error: `${label} must be at least ${min}`, correctedValue: min }
  }

  // Check max bound
  if (max !== undefined && value > max) {
    return { valid: false, error: `${label} must be at most ${max}`, correctedValue: max }
  }

  // Check if value aligns with step (optional - just warn)
  if (step !== undefined && step > 0) {
    const remainder = (value - (min || 0)) % step
    if (remainder !== 0 && Math.abs(remainder) > 0.0001) {
      // Allow small floating point errors
      const corrected = Math.round((value - (min || 0)) / step) * step + (min || 0)
      return { valid: true, error: null, correctedValue: corrected }
    }
  }

  return { valid: true, error: null, correctedValue: value }
}

/**
 * Validates grid size.
 * @param {number} size - The grid size to validate
 * @returns {{ valid: boolean, error: string | null }}
 */
export function validateGridSize(size) {
  if (typeof size !== 'number' || isNaN(size)) {
    return { valid: false, error: 'Grid size must be a number' }
  }

  if (!Number.isInteger(size)) {
    return { valid: false, error: 'Grid size must be a whole number' }
  }

  if (size < 5) {
    return { valid: false, error: 'Grid size must be at least 5' }
  }

  if (size > 50) {
    return { valid: false, error: 'Grid size must be at most 50' }
  }

  return { valid: true, error: null }
}

/**
 * Sanitizes a string for safe use in filenames.
 * @param {string} str - The string to sanitize
 * @returns {string}
 */
export function sanitizeFilename(str) {
  if (!str || typeof str !== 'string') return 'unnamed'
  return str
    .trim()
    .replace(/[^a-zA-Z0-9\-_]/g, '_')
    .substring(0, 50) || 'unnamed'
}
