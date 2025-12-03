"""
Safe formula evaluation module for reward calculations.

This module provides a secure way to evaluate mathematical formulas
without using Python's eval() function. It supports basic arithmetic
operations and variable substitution while preventing arbitrary code execution.
"""

import re
import ast
import operator
from typing import Dict, Any, Optional, Union


class SafeFormulaError(Exception):
    """Exception raised for formula parsing or evaluation errors."""
    pass


class SafeFormulaEvaluator:
    """
    A safe mathematical formula evaluator that prevents arbitrary code execution.

    Supported operations:
        - Addition (+)
        - Subtraction (-)
        - Multiplication (*)
        - Division (/)
        - Exponentiation (**)
        - Unary minus (-)
        - Parentheses for grouping

    Example usage:
        >>> evaluator = SafeFormulaEvaluator()
        >>> variables = {"snake_length": 5, "score": 10}
        >>> result = evaluator.evaluate("1.0 + 0.1 * snake_length", variables)
        >>> print(result)  # 1.5
    """

    # Operators mapped to their functions
    BINARY_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    UNARY_OPERATORS = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Valid variable names pattern (alphanumeric and underscore)
    VALID_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    def __init__(self, max_formula_length: int = 500):
        """
        Initialize the evaluator.

        Args:
            max_formula_length: Maximum allowed formula string length
        """
        self.max_formula_length = max_formula_length

    def validate_formula(self, formula: str) -> bool:
        """
        Validate a formula string without evaluating it.

        Args:
            formula: The formula string to validate

        Returns:
            True if the formula is valid

        Raises:
            SafeFormulaError: If the formula is invalid
        """
        if not isinstance(formula, str):
            raise SafeFormulaError("Formula must be a string")

        if len(formula) > self.max_formula_length:
            raise SafeFormulaError(
                f"Formula exceeds maximum length of {self.max_formula_length} characters"
            )

        if not formula.strip():
            raise SafeFormulaError("Formula cannot be empty")

        try:
            tree = ast.parse(formula, mode='eval')
        except SyntaxError as e:
            raise SafeFormulaError(f"Invalid formula syntax: {e}")

        # Validate all nodes in the AST
        self._validate_node(tree.body)

        return True

    def _validate_node(self, node: ast.AST) -> None:
        """
        Recursively validate an AST node.

        Args:
            node: The AST node to validate

        Raises:
            SafeFormulaError: If an unsafe node type is detected
        """
        if isinstance(node, ast.Constant):
            # Allow numeric constants only
            if not isinstance(node.value, (int, float)):
                raise SafeFormulaError(
                    f"Only numeric constants allowed, got: {type(node.value).__name__}"
                )
        elif isinstance(node, ast.Name):
            # Validate variable name format
            if not self.VALID_NAME_PATTERN.match(node.id):
                raise SafeFormulaError(f"Invalid variable name: {node.id}")
        elif isinstance(node, ast.BinOp):
            # Validate binary operation
            if type(node.op) not in self.BINARY_OPERATORS:
                raise SafeFormulaError(
                    f"Unsupported operator: {type(node.op).__name__}"
                )
            self._validate_node(node.left)
            self._validate_node(node.right)
        elif isinstance(node, ast.UnaryOp):
            # Validate unary operation
            if type(node.op) not in self.UNARY_OPERATORS:
                raise SafeFormulaError(
                    f"Unsupported unary operator: {type(node.op).__name__}"
                )
            self._validate_node(node.operand)
        else:
            # Reject all other node types (function calls, attribute access, etc.)
            raise SafeFormulaError(
                f"Unsupported expression type: {type(node).__name__}. "
                "Only arithmetic expressions with +, -, *, /, ** are allowed."
            )

    def get_variable_names(self, formula: str) -> set:
        """
        Extract all variable names used in a formula.

        Args:
            formula: The formula string

        Returns:
            Set of variable names used in the formula
        """
        self.validate_formula(formula)
        tree = ast.parse(formula, mode='eval')
        names = set()
        self._collect_names(tree.body, names)
        return names

    def _collect_names(self, node: ast.AST, names: set) -> None:
        """Recursively collect variable names from an AST."""
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.BinOp):
            self._collect_names(node.left, names)
            self._collect_names(node.right, names)
        elif isinstance(node, ast.UnaryOp):
            self._collect_names(node.operand, names)

    def evaluate(
        self,
        formula: str,
        variables: Dict[str, Union[int, float]]
    ) -> float:
        """
        Safely evaluate a mathematical formula with variable substitution.

        Args:
            formula: The formula string (e.g., "1.0 + 0.1 * snake_length")
            variables: Dictionary mapping variable names to numeric values

        Returns:
            The evaluated result as a float

        Raises:
            SafeFormulaError: If the formula is invalid or uses undefined variables
        """
        # Validate the formula structure
        self.validate_formula(formula)

        # Parse the formula
        tree = ast.parse(formula, mode='eval')

        # Check that all variables are defined
        used_vars = self.get_variable_names(formula)
        missing_vars = used_vars - set(variables.keys())
        if missing_vars:
            raise SafeFormulaError(
                f"Undefined variables: {', '.join(sorted(missing_vars))}"
            )

        # Evaluate the AST
        try:
            result = self._eval_node(tree.body, variables)
        except ZeroDivisionError:
            raise SafeFormulaError("Division by zero")
        except OverflowError:
            raise SafeFormulaError("Numeric overflow in calculation")

        return float(result)

    def _eval_node(
        self,
        node: ast.AST,
        variables: Dict[str, Union[int, float]]
    ) -> Union[int, float]:
        """
        Recursively evaluate an AST node.

        Args:
            node: The AST node to evaluate
            variables: Variable mapping

        Returns:
            The numeric result
        """
        if isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            return variables[node.id]

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            op_func = self.BINARY_OPERATORS[type(node.op)]
            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            op_func = self.UNARY_OPERATORS[type(node.op)]
            return op_func(operand)

        # Should not reach here due to validation
        raise SafeFormulaError(f"Unexpected node type: {type(node).__name__}")


# Module-level convenience function
_default_evaluator = SafeFormulaEvaluator()


def safe_eval(
    formula: str,
    variables: Dict[str, Union[int, float]]
) -> float:
    """
    Safely evaluate a mathematical formula with variable substitution.

    This is a convenience function using the default evaluator.

    Args:
        formula: The formula string (e.g., "1.0 + 0.1 * snake_length")
        variables: Dictionary mapping variable names to numeric values

    Returns:
        The evaluated result as a float

    Raises:
        SafeFormulaError: If the formula is invalid or uses undefined variables

    Example:
        >>> variables = {"snake_length": 5, "score": 10}
        >>> result = safe_eval("1.0 + 0.1 * snake_length", variables)
        >>> print(result)  # 1.5
    """
    return _default_evaluator.evaluate(formula, variables)


def validate_formula(formula: str) -> bool:
    """
    Validate a formula string without evaluating it.

    Args:
        formula: The formula string to validate

    Returns:
        True if the formula is valid

    Raises:
        SafeFormulaError: If the formula is invalid
    """
    return _default_evaluator.validate_formula(formula)


def get_formula_variables(formula: str) -> set:
    """
    Extract all variable names used in a formula.

    Args:
        formula: The formula string

    Returns:
        Set of variable names used in the formula
    """
    return _default_evaluator.get_variable_names(formula)
