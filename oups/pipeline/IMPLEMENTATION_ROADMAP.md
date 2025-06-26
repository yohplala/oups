# OUPS Memory-Aware Pipeline Implementation Roadmap

## **Proof of Concept Summary**

We've successfully demonstrated the core cascading execution pattern with:

1. **`pipeline_poc.py`** - Basic iteration-count triggered cascading
2. **`pipeline_poc_enhanced.py`** - Enhanced with buffer visualization
3. **`memory_pipeline_poc.py`** - Memory-triggered cascading with real data processing

## **Key Patterns Proven**

### ✅ **Cascading Trigger Pattern**
```
op1(iter_max=3) → op2(iter_max=2) → op3(iter_max=1)

Process sequence:
op1, op1, op1 → op2
op1, op1, op1 → op2 → op3 (full cascade, all buffers reset)
op1, op1, op1 → op2
...
```

### ✅ **Memory-Triggered Execution**
- Operations buffer data until memory limit reached
- When limit hit → process buffer → pass result to next operation
- Automatic buffer management and memory tracking

### ✅ **Registration & Chaining Pattern**
```python
@Pipeline.register_op()
def my_operation():
    # operation logic here
    pass

pipeline = Pipeline()
pipeline.op1(memory_limit_mb=50).op2(memory_limit_mb=100).op3(memory_limit_mb=200)
```

## **Extension to OUPS Architecture**

### **Phase 1: Core Memory-Aware Operations**

#### **1.1 Memory-Aware Streamz Extensions**
```python
# Extend streamz with memory-triggered operations
@Stream.register_api()
class memory_buffer(Stream):
    """Buffer with memory limit instead of count limit"""

    def __init__(self, upstream, memory_limit_mb, accumulate=True, **kwargs):
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.accumulate = accumulate
        self._buffer = []
        self._memory_usage = 0
        # ... implement memory tracking & flushing

@Stream.register_api()
class memory_accumulate(accumulate):
    """Accumulate with memory awareness"""

    def __init__(self, upstream, func, memory_limit_mb=None, accumulate=True, **kwargs):
        super().__init__(upstream, func, **kwargs)
        self.memory_limit = memory_limit_mb * 1024 * 1024 if memory_limit_mb else None
        self.accumulate_flag = accumulate
        # ... implement memory-triggered state flushing
```

#### **1.2 OUPS-Specific Operations**
```python
@Stream.register_api()
class aggstream_memory(memory_buffer):
    """Memory-aware aggregation for OUPS"""

    def __init__(self, upstream, memory_limit_mb, agg_configs, **kwargs):
        super().__init__(upstream, memory_limit_mb, **kwargs)
        self.agg_configs = agg_configs  # Multiple aggregation configurations

    def _process_buffer(self):
        """Process accumulated data through multiple aggregation configs"""
        results = {}
        for config_name, config in self.agg_configs.items():
            results[config_name] = self._apply_aggregation(self._buffer, config)
        return results

@Stream.register_api()
class joinstream_memory(memory_buffer):
    """Memory-aware join operations for OUPS"""

    def __init__(self, *upstreams, memory_limit_mb, join_config, **kwargs):
        super().__init__(upstreams, memory_limit_mb, **kwargs)
        self.join_config = join_config

    def _process_buffer(self):
        """Process multiple input streams through join logic"""
        return self._apply_join(self._buffers, self.join_config)
```

### **Phase 2: OUPS Pipeline Integration**

#### **2.1 OUPS Memory Pipeline Builder**
```python
class OUPSMemoryPipeline:
    """High-level builder for OUPS memory-aware data pipelines"""

    def __init__(self, base_memory_limit_mb=100):
        self.base_memory_limit = base_memory_limit_mb
        self.operations = []

    def add_aggstream(self, agg_configs, memory_limit_mb=None, accumulate=True):
        """Add aggregation stream with memory management"""
        memory_limit = memory_limit_mb or self.base_memory_limit

        def _add_to_pipeline(upstream):
            return upstream.aggstream_memory(
                memory_limit_mb=memory_limit,
                agg_configs=agg_configs,
                accumulate=accumulate
            )
        self.operations.append(_add_to_pipeline)
        return self

    def add_joinstream(self, join_config, memory_limit_mb=None, accumulate=True):
        """Add join stream with memory management"""
        memory_limit = memory_limit_mb or self.base_memory_limit

        def _add_to_pipeline(upstream):
            return upstream.joinstream_memory(
                memory_limit_mb=memory_limit,
                join_config=join_config,
                accumulate=accumulate
            )
        self.operations.append(_add_to_pipeline)
        return self

    def add_store_writer(self, store_config, memory_limit_mb=None):
        """Add final store writer operation"""
        memory_limit = memory_limit_mb or self.base_memory_limit

        def _add_to_pipeline(upstream):
            return upstream.memory_buffer(
                memory_limit_mb=memory_limit,
                accumulate=False  # Write operation doesn't accumulate
            ).sink(lambda data: self._write_to_store(data, store_config))
        self.operations.append(_add_to_pipeline)
        return self

    def build(self, source_stream):
        """Build the complete pipeline"""
        stream = source_stream
        for op in self.operations:
            stream = op(stream)
        return stream

    def _write_to_store(self, data, store_config):
        """Write processed data to OUPS store"""
        # Integration with existing OUPS store functionality
        pass
```

#### **2.2 Usage Pattern for OUPS**
```python
# Real OUPS usage example
from oups.aggstream import AggStream
from streamz import Stream

# Create source stream from OUPS data
source = Stream()

# Define aggregation configurations
hourly_agg = {
    'time_grouper': pd.Grouper(freq='H'),
    'agg_funcs': {'value': 'mean', 'count': 'sum'}
}

daily_agg = {
    'time_grouper': pd.Grouper(freq='D'),
    'agg_funcs': {'value': 'mean', 'count': 'sum'}
}

# Build memory-aware pipeline
pipeline = (OUPSMemoryPipeline(base_memory_limit_mb=200)
            .add_aggstream(
                agg_configs={
                    'hourly': hourly_agg,
                    'daily': daily_agg
                },
                memory_limit_mb=150,
                accumulate=True
            )
            .add_aggstream(
                agg_configs={'weekly': weekly_agg},
                memory_limit_mb=100,
                accumulate=True
            )
            .add_store_writer(
                store_config={'path': '/data/aggregated'},
                memory_limit_mb=50
            ))

# Execute pipeline
memory_aware_stream = pipeline.build(source)

# Feed data (this would be automatic from OUPS data sources)
for data_chunk in oups_data_iterator():
    source.emit(data_chunk)
```

### **Phase 3: Advanced Features**

#### **3.1 Memory Pressure Handling**
```python
class MemoryPressureManager:
    """Handle system memory pressure gracefully"""

    def __init__(self, max_system_memory_pct=80):
        self.max_memory_pct = max_system_memory_pct

    def should_trigger_flush(self, current_usage, limit):
        """Check if we should flush early due to system pressure"""
        system_memory_pct = self._get_system_memory_usage()
        if system_memory_pct > self.max_memory_pct:
            return True  # Flush early
        return current_usage >= limit

    def _get_system_memory_usage(self):
        """Get current system memory usage percentage"""
        import psutil
        return psutil.virtual_memory().percent
```

#### **3.2 Dynamic Memory Adjustment**
```python
class AdaptiveMemoryManager:
    """Dynamically adjust memory limits based on data characteristics"""

    def __init__(self):
        self.data_size_history = deque(maxlen=100)

    def suggest_memory_limit(self, operation_type, data_sample):
        """Suggest optimal memory limit based on data characteristics"""
        sample_size = sys.getsizeof(data_sample)
        self.data_size_history.append(sample_size)

        avg_size = sum(self.data_size_history) / len(self.data_size_history)

        # Suggest limit based on operation type and data size patterns
        if operation_type == 'aggregation':
            return avg_size * 1000  # Buffer ~1000 records for aggregation
        elif operation_type == 'join':
            return avg_size * 5000   # Larger buffer for join operations
        else:
            return avg_size * 500    # Default buffer size
```

## **Implementation Priority**

### **Immediate (Week 1-2)**
1. ✅ **Basic Pipeline POC** - COMPLETED
2. ✅ **Memory-triggered POC** - COMPLETED
3. **Streamz memory_buffer extension** - Core memory-aware buffering
4. **Basic OUPS integration** - Connect to existing aggstream/joinstream

### **Short-term (Week 3-4)**
1. **Memory-aware accumulate operations** - Stateful operations with memory limits
2. **OUPS pipeline builder** - High-level API for building pipelines
3. **Store integration** - Connect to existing OUPS store functionality
4. **Basic memory monitoring** - Track memory usage and trigger flushing

### **Medium-term (Month 2)**
1. **Advanced memory management** - Pressure handling, adaptive limits
2. **Multi-stream coordination** - Handle complex pipeline topologies
3. **Performance optimization** - Efficient memory tracking for large DataFrames
4. **Comprehensive testing** - Memory leak detection, edge cases

### **Long-term (Month 3+)**
1. **Distributed memory management** - Coordinate memory across multiple workers
2. **Advanced analytics** - Memory usage patterns, optimization suggestions
3. **Integration with Dask** - Scale memory-aware operations across clusters
4. **Production monitoring** - Memory dashboard, alerting system

## **Key Benefits of This Approach**

1. **Proven Pattern** - POCs demonstrate the core execution model works
2. **Streamz Compatible** - Leverages existing streamz patterns and ecosystem
3. **OUPS Integration** - Natural extension of existing aggstream/joinstream APIs
4. **Memory Safety** - Prevents OOM errors with configurable limits
5. **Flexible Control** - `accumulate=True/False` flag for operation control
6. **Scalable Design** - Can extend to distributed systems later

## **Next Steps**

1. **Review POCs** - Confirm the execution pattern matches your requirements
2. **Define OUPS API** - Specify exact interfaces for aggstream_memory, etc.
3. **Implement Core** - Start with memory_buffer and basic OUPS integration
4. **Test Integration** - Validate with real OUPS data and use cases
5. **Iterate & Optimize** - Refine based on performance and usability feedback

The foundation is solid - the POCs prove the cascading execution pattern works exactly as designed! 🚀
