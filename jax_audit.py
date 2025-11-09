#!/usr/bin/env python3
"""
JAX Dependencies Audit Script for voiage v0.3.0

This script performs a comprehensive audit of JAX dependencies including:
- JAX availability and version
- JAX device detection (CPU/GPU)
- JAX functionality testing
- Performance characteristics
- Compatibility with current system

Usage:
    python jax_audit.py
"""

import sys
import os
import subprocess
import importlib
import warnings
from typing import Dict, List, Tuple, Any, Optional

def run_command(cmd: str) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Command timed out"
    except Exception as e:
        return 1, "", str(e)

def check_jax_installation() -> Dict[str, Any]:
    """Check JAX installation and basic information."""
    print("🔍 JAX Installation Audit")
    print("=" * 40)
    
    audit_results = {
        "jax_available": False,
        "jax_version": None,
        "jaxlib_available": False,
        "jaxlib_version": None,
        "jax_numpy_available": False,
        "error_message": None
    }
    
    try:
        # Try to import JAX
        import jax
        import jax.numpy as jnp
        
        audit_results["jax_available"] = True
        audit_results["jax_version"] = jax.__version__
        audit_results["jax_numpy_available"] = True
        
        print(f"✅ JAX Version: {jax.__version__}")
        
        # Check jaxlib
        try:
            import jaxlib
            audit_results["jaxlib_available"] = True
            audit_results["jaxlib_version"] = jaxlib.__version__
            print(f"✅ jaxlib Version: {jaxlib.__version__}")
        except ImportError:
            print("⚠️  jaxlib not available")
            
    except ImportError as e:
        audit_results["error_message"] = str(e)
        print(f"❌ JAX not available: {e}")
        return audit_results
    except Exception as e:
        audit_results["error_message"] = str(e)
        print(f"❌ Error importing JAX: {e}")
        return audit_results
    
    return audit_results

def check_jax_devices() -> Dict[str, Any]:
    """Check available JAX devices (CPU/GPU/TPU)."""
    print("\n🖥️  JAX Device Detection")
    print("=" * 40)
    
    device_info = {
        "devices": [],
        "device_count": 0,
        "cpu_device": None,
        "gpu_devices": [],
        "tpu_devices": [],
        "default_device": None,
        "error": None
    }
    
    try:
        import jax
        import jax.random as jrandom
        
        # Get all devices
        devices = jax.devices()
        device_info["devices"] = [str(d) for d in devices]
        device_info["device_count"] = len(devices)
        
        # Categorize devices
        for device in devices:
            device_str = str(device)
            device_info["devices"].append(device_str)
            
            if "cpu" in device_str.lower():
                device_info["cpu_device"] = device_str
                print(f"💻 CPU Device: {device_str}")
            elif "gpu" in device_str.lower() or "cuda" in device_str.lower():
                device_info["gpu_devices"].append(device_str)
                print(f"🖥️  GPU Device: {device_str}")
            elif "tpu" in device_str.lower():
                device_info["tpu_devices"].append(device_str)
                print(f"⚡ TPU Device: {device_str}")
        
        # Get default device
        device_info["default_device"] = str(jax.devices()[0])
        print(f"🏁 Default Device: {device_info['default_device']}")
        print(f"📊 Total Devices: {device_info['device_count']}")
        
    except Exception as e:
        device_info["error"] = str(e)
        print(f"❌ Error detecting devices: {e}")
    
    return device_info

def check_jax_functionality() -> Dict[str, Any]:
    """Test basic JAX functionality."""
    print("\n🧪 JAX Functionality Testing")
    print("=" * 40)
    
    test_results = {
        "basic_operations": False,
        "array_creation": False,
        "math_operations": False,
        "gradients": False,
        "jit_compilation": False,
        "device_placement": False,
        "gpu_memory": False,
        "errors": []
    }
    
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        
        # Test 1: Basic array operations
        try:
            a = jnp.array([1, 2, 3, 4])
            b = jnp.array([5, 6, 7, 8])
            c = a + b
            test_results["basic_operations"] = True
            print("✅ Basic array operations: PASS")
        except Exception as e:
            test_results["errors"].append(f"Basic operations: {e}")
            print(f"❌ Basic array operations: FAIL - {e}")
        
        # Test 2: Array creation
        try:
            zeros = jnp.zeros((100, 100))
            ones = jnp.ones((50, 50))
            random = jrandom.normal(jrandom.key(42), (25, 25))
            test_results["array_creation"] = True
            print("✅ Array creation: PASS")
        except Exception as e:
            test_results["errors"].append(f"Array creation: {e}")
            print(f"❌ Array creation: FAIL - {e}")
        
        # Test 3: Mathematical operations
        try:
            x = jnp.array([1.0, 2.0, 3.0, 4.0])
            y = jnp.exp(x) * jnp.sin(x)
            z = jnp.sum(jnp.sqrt(jnp.abs(y)))
            test_results["math_operations"] = True
            print("✅ Mathematical operations: PASS")
        except Exception as e:
            test_results["errors"].append(f"Math operations: {e}")
            print(f"❌ Mathematical operations: FAIL - {e}")
        
        # Test 4: Gradient computation
        try:
            def simple_function(x):
                return jnp.sum(x ** 2)
            
            grad_fn = jax.grad(simple_function)
            result = grad_fn(jnp.array([1.0, 2.0, 3.0]))
            test_results["gradients"] = True
            print("✅ Gradient computation: PASS")
        except Exception as e:
            test_results["errors"].append(f"Gradient computation: {e}")
            print(f"❌ Gradient computation: FAIL - {e}")
        
        # Test 5: JIT compilation
        try:
            @jax.jit
            def jitted_function(x):
                return jnp.sum(x ** 2 + 1)
            
            x = jnp.array([1.0, 2.0, 3.0])
            result = jitted_function(x)
            test_results["jit_compilation"] = True
            print("✅ JIT compilation: PASS")
        except Exception as e:
            test_results["errors"].append(f"JIT compilation: {e}")
            print(f"❌ JIT compilation: FAIL - {e}")
        
        # Test 6: Device placement
        try:
            a = jnp.array([1, 2, 3, 4])
            device_name = a.device()
            test_results["device_placement"] = True
            print(f"✅ Device placement: PASS (device: {device_name})")
        except Exception as e:
            test_results["errors"].append(f"Device placement: {e}")
            print(f"❌ Device placement: FAIL - {e}")
        
        # Test 7: GPU memory check (if GPU available)
        try:
            if "gpu" in [d.device_kind.lower() for d in jax.devices()]:
                import jax.lib.xla_bridge as xla_bridge
                client = xla_bridge.get_backend().client
                if hasattr(client, 'memory_stats'):
                    stats = client.memory_stats()
                    test_results["gpu_memory"] = True
                    print("✅ GPU memory stats: Available")
                else:
                    test_results["gpu_memory"] = False
                    print("⚠️  GPU memory stats: Not available")
            else:
                test_results["gpu_memory"] = None
                print("ℹ️  GPU memory stats: No GPU available")
        except Exception as e:
            test_results["errors"].append(f"GPU memory: {e}")
            print(f"❌ GPU memory check: FAIL - {e}")
        
    except Exception as e:
        test_results["errors"].append(f"General JAX test error: {e}")
        print(f"❌ General JAX test error: {e}")
    
    return test_results

def check_jax_performance() -> Dict[str, Any]:
    """Test JAX performance characteristics."""
    print("\n🚀 JAX Performance Testing")
    print("=" * 40)
    
    perf_results = {
        "cpu_performance": None,
        "gpu_performance": None,
        "compilation_warmup": None,
        "memory_usage": None,
        "errors": []
    }
    
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jrandom
        import time
        
        # Test CPU performance
        try:
            start_time = time.perf_counter()
            for _ in range(100):
                a = jrandom.normal(jrandom.key(42), (1000, 1000))
                b = jrandom.normal(jrandom.key(43), (1000, 1000))
                c = jnp.dot(a, b)
            end_time = time.perf_counter()
            
            perf_results["cpu_performance"] = end_time - start_time
            print(f"✅ CPU matrix operations (100 iterations): {end_time - start_time:.3f}s")
        except Exception as e:
            perf_results["errors"].append(f"CPU performance: {e}")
            print(f"❌ CPU performance test: FAIL - {e}")
        
        # Test GPU performance (if available)
        try:
            if "gpu" in [d.device_kind.lower() for d in jax.devices()]:
                start_time = time.perf_counter()
                for _ in range(100):
                    with jax.default_device(jax.devices()[0]):  # Use first GPU
                        a = jrandom.normal(jrandom.key(42), (1000, 1000))
                        b = jrandom.normal(jrandom.key(43), (1000, 1000))
                        c = jnp.dot(a, b)
                end_time = time.perf_counter()
                
                perf_results["gpu_performance"] = end_time - start_time
                print(f"✅ GPU matrix operations (100 iterations): {end_time - start_time:.3f}s")
            else:
                print("ℹ️  GPU performance test: No GPU available")
        except Exception as e:
            perf_results["errors"].append(f"GPU performance: {e}")
            print(f"❌ GPU performance test: FAIL - {e}")
        
        # Test JIT compilation warmup
        try:
            @jax.jit
            def test_function(x):
                return jnp.sum(jnp.sin(x) + jnp.cos(x))
            
            # First call (compilation)
            start_time = time.perf_counter()
            result1 = test_function(jnp.array([1.0, 2.0, 3.0, 4.0]))
            first_call_time = time.perf_counter() - start_time
            
            # Second call (cached)
            start_time = time.perf_counter()
            result2 = test_function(jnp.array([5.0, 6.0, 7.0, 8.0]))
            cached_call_time = time.perf_counter() - start_time
            
            perf_results["compilation_warmup"] = {
                "first_call": first_call_time,
                "cached_call": cached_call_time,
                "speedup": first_call_time / cached_call_time if cached_call_time > 0 else 0
            }
            
            print(f"✅ JIT compilation test:")
            print(f"   First call (compilation): {first_call_time:.4f}s")
            print(f"   Cached call: {cached_call_time:.4f}s")
            print(f"   Speedup: {first_call_time / cached_call_time:.1f}x")
            
        except Exception as e:
            perf_results["errors"].append(f"JIT compilation test: {e}")
            print(f"❌ JIT compilation test: FAIL - {e}")
            
    except Exception as e:
        perf_results["errors"].append(f"General performance test error: {e}")
        print(f"❌ General performance test error: {e}")
    
    return perf_results

def check_system_compatibility() -> Dict[str, Any]:
    """Check system compatibility for JAX."""
    print("\n🖥️  System Compatibility Check")
    print("=" * 40)
    
    compat_results = {
        "python_version": sys.version,
        "platform": sys.platform,
        "architecture": None,
        "cuda_version": None,
        "cudnn_version": None,
        "gcc_version": None,
        "errors": []
    }
    
    try:
        # Python version
        compat_results["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"🐍 Python Version: {compat_results['python_version']}")
        
        # Platform
        compat_results["platform"] = sys.platform
        print(f"💻 Platform: {compat_results['platform']}")
        
        # Architecture
        try:
            result = run_command("uname -m")
            if result[0] == 0:
                compat_results["architecture"] = result[1].strip()
                print(f"🏗️  Architecture: {compat_results['architecture']}")
        except Exception:
            pass  # Ignore if command fails
        
        # CUDA version (if available)
        try:
            result = run_command("nvcc --version 2>/dev/null")
            if result[0] == 0:
                version_line = [line for line in result[1].split('\n') if 'release' in line]
                if version_line:
                    compat_results["cuda_version"] = version_line[0].strip()
                    print(f"🚀 CUDA Version: Found")
        except Exception:
            pass  # Ignore if command fails
        
        # Check for CUDA through nvidia-smi
        try:
            result = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null")
            if result[0] == 0:
                compat_results["cuda_version"] = f"Driver: {result[1].strip()}"
                print(f"🚀 CUDA Version: {compat_results['cuda_version']}")
            else:
                print("🚀 CUDA Version: Not available")
        except Exception:
    # Ignore if CUDA not available
            print("🚀 CUDA Version: Not available")
        
        # GCC version
        try:
            result = run_command("gcc --version | head -1")
            if result[0] == 0:
                compat_results["gcc_version"] = result[1].strip()
                print(f"🔧 GCC Version: {compat_results['gcc_version']}")
        except:
            pass
            
    except Exception as e:
        compat_results["errors"].append(f"System compatibility check: {e}")
        print(f"❌ System compatibility check: {e}")
    
    return compat_results

def generate_recommendations(audit_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on audit results."""
    recommendations = []
    
    # JAX availability
    if not audit_data["jax"]["jax_available"]:
        recommendations.append("🔧 Install JAX: pip install jax jaxlib")
        recommendations.append("💡 For CPU-only: pip install jax[cpu]")
        recommendations.append("💡 For GPU support: pip install jax[cuda]")
    
    # JAX version
    if audit_data["jax"]["jax_available"]:
        jax_version = audit_data["jax"]["jax_version"]
        if jax_version:
            try:
                major, minor = map(int, jax_version.split('.')[:2])
                if major < 0 or (major == 0 and minor < 4):
                    recommendations.append(f"⚠️  JAX version {jax_version} is old. Consider updating to latest version.")
            except:
                pass
    
    # GPU availability
    gpu_count = len(audit_data["devices"]["gpu_devices"])
    if gpu_count == 0:
        recommendations.append("💡 No GPU detected. JAX will use CPU for all operations.")
        recommendations.append("💡 Consider installing CUDA for GPU acceleration: pip install jax[cuda]")
    else:
        recommendations.append(f"✅ {gpu_count} GPU(s) detected. JAX GPU acceleration is available!")
        recommendations.append("💡 Use JAX GPU backend for performance-critical computations")
    
    # Functionality issues
    if audit_data["functionality"]["errors"]:
        recommendations.append("⚠️  Some JAX functionality tests failed. Check JAX installation.")
    
    # Performance recommendations
    if audit_data["performance"]["cpu_performance"] and audit_data["performance"]["cpu_performance"] > 10:
        recommendations.append("⚠️  JAX CPU performance is slower than expected. Consider GPU acceleration.")
    
    return recommendations

def main():
    """Main audit function."""
    print("🔍 JAX Dependencies Audit for voiage v0.3.0")
    print("=" * 50)
    
    audit_data = {
        "jax": check_jax_installation(),
        "devices": check_jax_devices(),
        "functionality": check_jax_functionality(),
        "performance": check_jax_performance(),
        "compatibility": check_system_compatibility()
    }
    
    # Generate recommendations
    recommendations = generate_recommendations(audit_data)
    
    # Print summary
    print("\n📊 AUDIT SUMMARY")
    print("=" * 50)
    
    if audit_data["jax"]["jax_available"]:
        print("✅ JAX is available and functional")
        print(f"   Version: {audit_data['jax']['jax_version']}")
        print(f"   Devices: {audit_data['devices']['device_count']} total")
        print(f"   CPU: {len(audit_data['devices']['cpu_device'])} device(s)")
        print(f"   GPU: {len(audit_data['devices']['gpu_devices'])} device(s)")
    else:
        print("❌ JAX is not available")
        print("   Recommendation: Install JAX using pip install jax jaxlib")
    
    # Print recommendations
    if recommendations:
        print("\n💡 RECOMMENDATIONS")
        print("=" * 50)
        for rec in recommendations:
            print(rec)
    
    # Save results
    import json
    with open("jax_audit_results.json", "w") as f:
        json.dump(audit_data, f, indent=2, default=str)
    
    print(f"\n💾 Audit results saved to jax_audit_results.json")
    print(f"🚀 JAX audit completed!")
    
    return audit_data

if __name__ == "__main__":
    main()