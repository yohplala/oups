#!/usr/bin/env python3
"""
Basic Pipeline implementation with cascading operation triggers.

This module provides a minimal proof of concept for iteration-based pipeline
operations that trigger downstream processing when limits are reached.

Created on Wed Dec  1 18:35:00 2021.
@author: yoh

"""


class Pipeline:
    """
    Minimal Pipeline implementation with cascading operation triggers.

    Each operation runs up to iter_max times before triggering the next operation once.
    When an operation reaches its limit, it resets its counter and triggers the next op.

    """

    _registered_ops = {}

    def __init__(self):
        """
        Initialize the pipeline.

        Creates empty operation and state tracking lists.

        """
        self.operations = []  # List of operation definitions
        self.op_states = []  # Track current count for each operation

    @classmethod
    def register_op(cls, func=None):
        """
        Register operations in the pipeline.

        This decorator allows functions to be registered as pipeline operations
        that can be chained together with iteration limits.

        Parameters
        ----------
        func : callable, optional
            The function to register, by default None

        Returns
        -------
        callable
            The registered function or decorator

        Examples
        --------
        >>> @Pipeline.register_op()
        ... def my_op():
        ...     print("my_op")

        >>> @Pipeline.register_op
        ... def my_op():
        ...     print("my_op")

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

        When you call pipeline.op1(iter_max=3), this creates a pipeline step.

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

        Always starts with the first operation and cascades through the pipeline based
        on iteration limits.

        """
        if not self.operations:
            return

        # Always try to execute the first operation
        self._execute_operation(0)

    def _execute_operation(self, op_index):
        """
        Execute operation at given index and handle cascading.

        Parameters
        ----------
        op_index : int
            Index of operation to execute

        """
        if op_index >= len(self.operations):
            return

        op = self.operations[op_index]

        # Execute the operation
        op["func"]()
        self.op_states[op_index] += 1

        # Check if this operation has reached its limit
        if self.op_states[op_index] >= op["iter_max"]:
            # Reset this operation's counter (buffer is now "empty")
            self.op_states[op_index] = 0

            # Trigger the next operation once
            if op_index + 1 < len(self.operations):
                self._execute_operation(op_index + 1)

    def get_state(self):
        """
        Get current state of all operations for debugging.

        Returns
        -------
        list of tuple
            List of (op_name, current_count, iter_max) tuples

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
        print("op1")

    @Pipeline.register_op()
    def op2():
        """
        Second test operation for demonstration.
        """
        print("op2")

    @Pipeline.register_op  # equivalent to write
    def op3():
        """
        Third test operation for demonstration (write operation).
        """
        print("op3")

    # Create pipeline
    pipeline = Pipeline()

    # Configure pipeline: op1 runs 3 times before triggering op2,
    # op2 runs 2 times before triggering op3
    pipeline.op1(iter_max=3).op2(iter_max=2).op3(iter_max=1)

    print("Pipeline configured:")
    print("- op1: runs 3 times before triggering op2")
    print("- op2: runs 2 times before triggering op3")
    print("- op3: runs 1 time (write operation)")
    print("\nExecution pattern:")
    print("=" * 50)

    # Run pipeline for 20 iterations
    for i in range(20):
        print(f"Process {i+1:2d}: ", end="")
        pipeline.process()

        # Show state every few iterations for clarity
        if i % 6 == 5:  # Every 6 iterations
            state = pipeline.get_state()
            print(f"    State: {state}")
            print("-" * 30)
