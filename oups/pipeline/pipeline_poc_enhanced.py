#!/usr/bin/env python3
"""
Enhanced Pipeline implementation with detailed buffer visualization.

This module provides an enhanced proof of concept for iteration-based pipeline
operations with visual buffer state tracking and detailed execution logging.

Created on Wed Dec  1 18:35:00 2021.
@author: yoh

"""


class Pipeline:
    """
    Enhanced Pipeline implementation with detailed logging of buffer states.

    Each operation runs up to iter_max times before triggering the next operation once.
    When an operation reaches its limit, it resets its counter and triggers the next op.

    """

    _registered_ops = {}

    def __init__(self, verbose=False):
        """
        Initialize the enhanced pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Enable verbose logging with buffer state visualization, by default False

        """
        self.operations = []  # List of operation definitions
        self.op_states = []  # Track current count for each operation
        self.verbose = verbose
        self.total_processes = 0

    @classmethod
    def register_op(cls, func=None):
        """
        Register operations in the pipeline.

        This decorator allows functions to be registered as pipeline operations
        with enhanced visualization capabilities.

        Parameters
        ----------
        func : callable, optional
            The function to register, by default None

        Returns
        -------
        callable
            The registered function or decorator

        """

        def decorator(f):
            cls._registered_ops[f.__name__] = f
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def __getattr__(self, name):
        """
        Dynamic method creation for registered operations.
        """
        if name in self._registered_ops:

            def op_builder(iter_max=1):
                self.operations.append(
                    {
                        "name": name,
                        "func": self._registered_ops[name],
                        "iter_max": iter_max,
                    },
                )
                self.op_states.append(0)  # Initialize count to 0
                return self  # Allow chaining

            return op_builder
        raise AttributeError(f"Operation '{name}' not registered")

    def process(self):
        """
        Execute one step of the pipeline.
        """
        if not self.operations:
            return

        self.total_processes += 1

        if self.verbose:
            print(f"\n--- Process {self.total_processes} ---")
            self.print_buffer_states("Before:")

        # Always try to execute the first operation
        self._execute_operation(0)

        if self.verbose:
            self.print_buffer_states("After:")

    def _execute_operation(self, op_index):
        """
        Execute operation at given index and handle cascading.
        """
        if op_index >= len(self.operations):
            return

        op = self.operations[op_index]

        # Execute the operation
        print(f"    {'  ' * op_index}→ {op['name']}")
        op["func"]()
        self.op_states[op_index] += 1

        # Check if this operation has reached its limit
        if self.op_states[op_index] >= op["iter_max"]:
            print(
                f"    {'  ' * op_index}  └─ {op['name']} buffer full! Triggering next operation...",
            )
            # Reset this operation's counter (buffer is now "empty")
            self.op_states[op_index] = 0

            # Trigger the next operation once
            if op_index + 1 < len(self.operations):
                self._execute_operation(op_index + 1)

    def print_buffer_states(self, prefix=""):
        """
        Print current buffer states for all operations.
        """
        print(f"  {prefix} Buffer states:")
        for i, op in enumerate(self.operations):
            current = self.op_states[i]
            max_val = op["iter_max"]
            buffer_visual = "█" * current + "░" * (max_val - current)
            print(f"    {op['name']}: [{buffer_visual}] {current}/{max_val}")

    def get_state(self):
        """
        Get current state of all operations for debugging.
        """
        return [
            (op["name"], self.op_states[i], op["iter_max"]) for i, op in enumerate(self.operations)
        ]


# Example usage and test
if __name__ == "__main__":

    @Pipeline.register_op()
    def op1():
        """
        First test operation for demonstration.
        """
        pass  # Just increment counter, actual work would be here

    @Pipeline.register_op()
    def op2():
        """
        Second test operation for demonstration.
        """
        pass  # Just increment counter, actual work would be here

    @Pipeline.register_op  # equivalent to write
    def op3():
        """
        Third test operation for demonstration (write operation).
        """
        pass  # Just increment counter, actual work would be here

    # Create pipeline with verbose output
    pipeline = Pipeline(verbose=True)

    # Configure pipeline: op1 runs 3 times before triggering op2,
    # op2 runs 2 times before triggering op3
    pipeline.op1(iter_max=3).op2(iter_max=2).op3(iter_max=1)

    print("Enhanced Pipeline Demonstration")
    print("=" * 50)
    print("Configuration:")
    print("- op1: buffer size 3 (runs 3 times before triggering op2)")
    print("- op2: buffer size 2 (runs 2 times before triggering op3)")
    print("- op3: buffer size 1 (write operation, runs once)")
    print("\nLegend: █ = filled buffer slot, ░ = empty buffer slot")
    print("=" * 50)

    # Run pipeline for 10 iterations to show the pattern
    for i in range(10):
        pipeline.process()

        # Add separator every few operations
        if i % 3 == 2:
            print("\n" + "-" * 30)

    print("\n" + "=" * 50)
    print("Pattern Summary:")
    print("1. op1 fills buffer (3 times) → triggers op2 once")
    print("2. When op2 buffer fills (2 times) → triggers op3 once")
    print("3. op3 runs once (write) → all buffers reset")
    print("4. Cycle repeats...")
