"""
Tests for server-side validation utilities.
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.validation import (
    validate_model_name,
    validate_grid_size,
    validate_hyperparameters,
    validate_filename
)


class TestValidateModelName:
    """Tests for model name validation."""

    def test_valid_name(self):
        """Test that valid names pass validation."""
        is_valid, error, sanitized = validate_model_name("MyModel")
        assert is_valid
        assert error is None
        assert sanitized == "MyModel"

    def test_valid_name_with_spaces(self):
        """Test that names with spaces are sanitized."""
        is_valid, error, sanitized = validate_model_name("My Model 1")
        assert is_valid
        assert error is None
        assert "_" in sanitized  # Spaces become underscores

    def test_valid_name_with_hyphens(self):
        """Test that names with hyphens work."""
        is_valid, error, sanitized = validate_model_name("my-model")
        assert is_valid
        assert sanitized == "my-model"

    def test_empty_name_fails(self):
        """Test that empty names fail validation."""
        is_valid, error, sanitized = validate_model_name("")
        assert not is_valid
        assert error is not None

    def test_none_name_fails(self):
        """Test that None names fail validation."""
        is_valid, error, sanitized = validate_model_name(None)
        assert not is_valid
        assert error is not None

    def test_whitespace_only_fails(self):
        """Test that whitespace-only names fail."""
        is_valid, error, sanitized = validate_model_name("   ")
        assert not is_valid

    def test_too_long_name_fails(self):
        """Test that names over 50 chars fail."""
        long_name = "a" * 51
        is_valid, error, sanitized = validate_model_name(long_name)
        assert not is_valid
        assert "50" in error

    def test_special_chars_fail(self):
        """Test that special characters fail."""
        is_valid, error, sanitized = validate_model_name("model@#$!")
        assert not is_valid


class TestValidateGridSize:
    """Tests for grid size validation."""

    def test_valid_grid_size(self):
        """Test that valid grid sizes pass."""
        is_valid, error, corrected = validate_grid_size(10)
        assert is_valid
        assert error is None
        assert corrected == 10

    def test_minimum_grid_size(self):
        """Test minimum grid size boundary."""
        is_valid, error, corrected = validate_grid_size(5)
        assert is_valid
        assert corrected == 5

    def test_maximum_grid_size(self):
        """Test maximum grid size boundary."""
        is_valid, error, corrected = validate_grid_size(50)
        assert is_valid
        assert corrected == 50

    def test_too_small_grid_size(self):
        """Test that grid size < 5 fails."""
        is_valid, error, corrected = validate_grid_size(4)
        assert not is_valid
        assert corrected == 5

    def test_too_large_grid_size(self):
        """Test that grid size > 50 fails."""
        is_valid, error, corrected = validate_grid_size(51)
        assert not is_valid
        assert corrected == 50

    def test_string_grid_size(self):
        """Test that string grid sizes are converted."""
        is_valid, error, corrected = validate_grid_size("10")
        assert is_valid
        assert corrected == 10

    def test_invalid_string_fails(self):
        """Test that non-numeric strings fail."""
        is_valid, error, corrected = validate_grid_size("abc")
        assert not is_valid


class TestValidateHyperparameters:
    """Tests for hyperparameter validation."""

    def test_valid_dqn_params(self):
        """Test that valid DQN params pass."""
        params = {
            'buffer': 10000,
            'batch': 128,
            'gamma': 0.9,
            'decay': 0.999
        }
        is_valid, errors, corrected = validate_hyperparameters(params, 'dqn')
        assert is_valid
        assert len(errors) == 0
        assert corrected['buffer'] == 10000

    def test_valid_ppo_params(self):
        """Test that valid PPO params pass."""
        params = {
            'buffer': 1000,
            'batch': 128,
            'gamma': 0.99,
            'decay': 1000,
            'epoch': 8
        }
        is_valid, errors, corrected = validate_hyperparameters(params, 'ppo')
        assert is_valid
        assert len(errors) == 0

    def test_missing_params_use_defaults(self):
        """Test that missing params get default values."""
        params = {}
        is_valid, errors, corrected = validate_hyperparameters(params, 'dqn')
        assert 'buffer' in corrected
        assert 'batch' in corrected
        assert 'gamma' in corrected
        assert 'decay' in corrected

    def test_out_of_range_gamma_corrected(self):
        """Test that gamma > 1 is corrected."""
        params = {'gamma': 1.5}
        is_valid, errors, corrected = validate_hyperparameters(params, 'dqn')
        assert not is_valid
        assert 'gamma' in errors
        assert corrected['gamma'] == 1.0

    def test_negative_buffer_corrected(self):
        """Test that negative buffer is corrected."""
        params = {'buffer': -100}
        is_valid, errors, corrected = validate_hyperparameters(params, 'dqn')
        assert not is_valid
        assert 'buffer' in errors
        assert corrected['buffer'] == 100  # Minimum


class TestValidateFilename:
    """Tests for filename validation."""

    def test_valid_filename(self):
        """Test that valid filenames pass."""
        is_valid, error = validate_filename("model_dqn_123456")
        assert is_valid
        assert error is None

    def test_path_traversal_blocked(self):
        """Test that path traversal attempts are blocked."""
        is_valid, error = validate_filename("../../../etc/passwd")
        assert not is_valid
        assert "Invalid" in error

    def test_forward_slash_blocked(self):
        """Test that forward slashes are blocked."""
        is_valid, error = validate_filename("path/to/file")
        assert not is_valid

    def test_backslash_blocked(self):
        """Test that backslashes are blocked."""
        is_valid, error = validate_filename("path\\to\\file")
        assert not is_valid

    def test_empty_filename_fails(self):
        """Test that empty filenames fail."""
        is_valid, error = validate_filename("")
        assert not is_valid

    def test_special_chars_fail(self):
        """Test that special characters fail."""
        is_valid, error = validate_filename("file@name.txt")
        assert not is_valid

    def test_too_long_filename_fails(self):
        """Test that filenames over 100 chars fail."""
        long_name = "a" * 101
        is_valid, error = validate_filename(long_name)
        assert not is_valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
