#!/usr/bin/env python3
"""
Memory-triggered Pipeline implementation for cascading data processing.

This module provides a proof of concept for memory-aware pipeline operations
that trigger downstream processing when memory limits are reached.

Created on Wed Dec  1 18:35:00 2021.
@author: yoh

"""

import sys


class MemoryPipeline:
    """
    Memory-triggered Pipeline implementation.

    Operations are triggered when memory usage reaches a limit instead of iteration
    count. This demonstrates how the concept could extend to real memory-aware
    processing.

    """

    _registered_ops = {}

    def __init__(self, verbose=False):
        """
        Initialize the memory pipeline.

        Parameters
        ----------
        verbose : bool, optional
            Enable verbose logging of pipeline operations, by default False

        """
        self.operations = []  # List of operation definitions
        self.buffers = []  # Actual data buffers for each operation
        self.memory_usage = []  # Memory usage tracking
        self.verbose = verbose
        self.total_processes = 0

    @classmethod
    def register_op(cls, func=None):
        """
        Register operations in the pipeline.

        This decorator allows functions to be registered as pipeline operations
        that can be chained together with memory limits.

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

            def op_builder(memory_limit_mb=1):
                self.operations.append(
                    {
                        "name": name,
                        "func": self._registered_ops[name],
                        "memory_limit": memory_limit_mb * 1024 * 1024,  # Convert to bytes
                    },
                )
                self.buffers.append([])  # Initialize empty buffer
                self.memory_usage.append(0)  # Initialize memory usage
                return self  # Allow chaining

            return op_builder
        raise AttributeError(f"Operation '{name}' not registered")

    def process(self, data):
        """
        Process one data element through the pipeline.

        Parameters
        ----------
        data : any
            The data element to process

        """
        if not self.operations:
            return

        self.total_processes += 1

        if self.verbose:
            print(f"\n--- Process {self.total_processes}: Processing {data} ---")
            self.print_memory_states("Before:")

        # Always try to add data to the first operation
        self._add_to_buffer(0, data)

        if self.verbose:
            self.print_memory_states("After:")

    def _add_to_buffer(self, op_index, data):
        """
        Add data to operation buffer and handle memory-triggered cascading.
        """
        if op_index >= len(self.operations):
            return

        op = self.operations[op_index]

        # Add data to buffer
        self.buffers[op_index].append(data)
        self.memory_usage[op_index] += self._get_memory_size(data)

        print(f"    {'  ' * op_index}→ {op['name']}: added {data}")

        # Check if memory limit reached
        if self.memory_usage[op_index] >= op["memory_limit"]:
            print(
                f"    {'  ' * op_index}  └─ {op['name']} memory limit reached! Processing buffer...",
            )

            # Process the buffer through the operation
            buffer_data = list(self.buffers[op_index])
            processed_data = op["func"](buffer_data)

            # Clear buffer and reset memory usage
            self.buffers[op_index].clear()
            self.memory_usage[op_index] = 0

            # If there's a next operation and we have processed data, pass it along
            if op_index + 1 < len(self.operations) and processed_data is not None:
                self._add_to_buffer(op_index + 1, processed_data)

    def _get_memory_size(self, obj):
        """
        Get approximate memory size of an object.

        In a real implementation, this would be more sophisticated.

        """
        return sys.getsizeof(obj)

    def print_memory_states(self, prefix=""):
        """
        Print current memory states for all operations.
        """
        print(f"  {prefix} Memory states:")
        for i, op in enumerate(self.operations):
            current_mb = self.memory_usage[i] / (1024 * 1024)
            limit_mb = op["memory_limit"] / (1024 * 1024)
            buffer_size = len(self.buffers[i])

            # Create visual representation
            if limit_mb > 0:
                usage_pct = min(100, int((current_mb / limit_mb) * 100))
                visual_filled = "█" * (usage_pct // 10)
                visual_empty = "░" * (10 - (usage_pct // 10))
                visual = f"[{visual_filled}{visual_empty}]"
            else:
                visual = "[          ]"

            print(
                f"    {op['name']}: {visual} {current_mb:.3f}/{limit_mb:.1f}MB ({buffer_size} items)",
            )

    def get_buffer_contents(self):
        """
        Get current buffer contents for debugging.
        """
        return {op["name"]: list(self.buffers[i]) for i, op in enumerate(self.operations)}


# Example operations that process data
@MemoryPipeline.register_op()
def filter_data(buffer):
    """Filter operation - removes even numbers."""
    filtered = [x for x in buffer if x % 2 != 0]
    print(f"      filter_data: processed {len(buffer)} items → {len(filtered)} items")
    return filtered


@MemoryPipeline.register_op()
def aggregate_data(buffer):
    """
    Aggregate data by summing all values.
    """
    if isinstance(buffer, list) and buffer:
        result = sum(buffer)
        print(f"      aggregate_data: summed {len(buffer)} items → {result}")
        return result
    return buffer


@MemoryPipeline.register_op()
def write_data(buffer):
    """
    Write data to storage as final output.
    """
    print(f"      write_data: WRITING {buffer} to storage")
    return None  # No further processing


# Example usage
if __name__ == "__main__":

    # Create memory-based pipeline
    pipeline = MemoryPipeline(verbose=True)

    # Configure pipeline with memory limits:
    # - filter_data: 0.001 MB limit (very small, triggers frequently)
    # - aggregate_data: 0.001 MB limit
    # - write_data: 0.001 MB limit
    pipeline.filter_data(memory_limit_mb=0.001).aggregate_data(memory_limit_mb=0.001).write_data(
        memory_limit_mb=0.001,
    )

    print("Memory-Based Pipeline Demonstration")
    print("=" * 60)
    print("Configuration:")
    print("- filter_data: removes even numbers, 0.001MB memory limit")
    print("- aggregate_data: sums values, 0.001MB memory limit")
    print("- write_data: outputs result, 0.001MB memory limit")
    print("\nProcessing sequence of numbers...")
    print("=" * 60)

    # Process a sequence of numbers through the pipeline
    test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    for i, data in enumerate(test_data):
        pipeline.process(data)

        # Add separator every few operations
        if i % 4 == 3:
            print("\n" + "-" * 40)

    print("\n" + "=" * 60)
    print("Memory-Based Pattern Summary:")
    print("1. Data accumulates in filter_data until memory limit")
    print("2. When limit reached → process buffer and pass to aggregate_data")
    print("3. aggregate_data accumulates until memory limit")
    print("4. When limit reached → process and pass to write_data")
    print("5. write_data outputs immediately (final operation)")
    print("\nThis demonstrates memory-triggered cascading execution!")
