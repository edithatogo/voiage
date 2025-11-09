
# GPU Acceleration Utilities for JAX Backend
import jax
import jax.numpy as jnp

class GpuAcceleration:
    """Utilities for GPU acceleration and memory management."""
    
    @staticmethod
    def detect_gpu():
        """Detect available GPU devices."""
        devices = jax.devices()
        gpu_devices = [device for device in devices if 'gpu' in device.device_kind.lower()]
        return gpu_devices
    
    @staticmethod
    def get_memory_info():
        """Get GPU memory information."""
        try:
            import jaxlib.xla_extension as xla
            # This is a placeholder - actual implementation would depend on JAX version
            return {
                'gpu_available': len(GpuAcceleration.detect_gpu()) > 0,
                'gpu_count': len(GpuAcceleration.detect_gpu()),
                'memory_info': 'Available via jax.lib.xla_bridge'
            }
        except Exception:
            return {
                'gpu_available': False,
                'gpu_count': 0,
                'memory_info': 'Unable to query memory info'
            }
    
    @staticmethod
    def optimize_for_gpu(data):
        """Optimize data layout for GPU processing."""
        # Ensure data is in optimal format for GPU
        if hasattr(data, 'device_buffer'):
            # JAX array - already optimized
            return data
        else:
            # Convert to JAX array with optimal dtype
            return jnp.asarray(data, dtype=jnp.float32)  # float32 often faster on GPU
    
    @staticmethod
    def memory_efficient_batch_process(data_batches, process_func, max_memory_mb=1000):
        """Process large datasets in memory-efficient batches."""
        results = []
        current_memory = 0
        
        for i, batch in enumerate(data_batches):
            batch_size_mb = batch.nbytes / (1024 * 1024)
            
            if current_memory + batch_size_mb > max_memory_mb:
                # Process current results to free memory
                print(f"Processing batch {i} to free memory")
                results = process_func(results)
                current_memory = 0
                
            # Add batch to current memory usage
            current_memory += batch_size_mb
            results.append(batch)
            
        # Process final results
        return process_func(results)
