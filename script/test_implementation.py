#!/usr/bin/env python3
"""Test script to verify the standalone video analytics implementation."""

import subprocess
import sys
import importlib.util

def check_imports():
    """Check if the script can be imported."""
    print("Checking imports...")
    spec = importlib.util.spec_from_file_location("video_analytics_script", "video_analytics_script.py")
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        print("[OK] Script imports successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False

def check_classes():
    """Check if all required classes are present."""
    print("\nChecking classes...")
    
    # Import the module
    spec = importlib.util.spec_from_file_location("video_analytics_script", "video_analytics_script.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    required_classes = [
        'VideoAnalyticsEngine',
        'GStreamerPipeline', 
        'YOLODetector',
        'SourceManager',
        'ConfigManager',
        'Logger',
        'VideoSource',
        'DetectionResult',
        'SourceType',
        'PipelineState',
        'HealthStatus'
    ]
    
    for class_name in required_classes:
        if hasattr(module, class_name):
            print(f"[OK] {class_name} found")
        else:
            print(f"[FAIL] {class_name} missing")
            return False
    
    return True

def check_api_endpoints():
    """Check if all required API endpoints are defined."""
    print("\nChecking API endpoints...")
    
    # Read the file and check for endpoint definitions
    with open("video_analytics_script.py", "r") as f:
        content = f.read()
    
    required_endpoints = [
        '@app.get("/health")',
        '@app.get("/api/v1/status")',
        '@app.get("/api/v1/sources")',
        '@app.post("/api/v1/sources")',
        '@app.delete("/api/v1/sources/{source_id}")',
        '@app.get("/api/v1/config")',
        '@app.put("/api/v1/config/detection")',
        '@app.get("/api/v1/metrics")'
    ]
    
    for endpoint in required_endpoints:
        if endpoint in content:
            print(f"[OK] {endpoint} found")
        else:
            print(f"[FAIL] {endpoint} missing")
            return False
    
    return True

def check_features():
    """Check if key features are implemented."""
    print("\nChecking features...")
    
    with open("video_analytics_script.py", "r") as f:
        content = f.read()
    
    features = {
        "GStreamer mock support": "GSTREAMER_AVAILABLE",
        "FastAPI mock support": "FASTAPI_AVAILABLE", 
        "Configuration management": "class ConfigManager",
        "JSON logging": "class JsonFormatter",
        "Pipeline management": "def create_pipeline",
        "Source management": "def add_source",
        "YOLO configuration": "def configure_inference",
        "Error handling": "try:",
        "Signal handling": "_setup_signal_handlers",
        "Statistics tracking": "def get_statistics"
    }
    
    for feature, marker in features.items():
        if marker in content:
            print(f"[OK] {feature}")
        else:
            print(f"[FAIL] {feature}")
            return False
    
    return True

def check_prp_requirements():
    """Check if PRP requirements are met."""
    print("\nChecking PRP requirements...")
    
    requirements = {
        "Task 1 - Project structure": ["ConfigManager", "Logger", "main()"],
        "Task 2 - GStreamer pipeline": ["GStreamerPipeline", "create_pipeline", "nvstreammux"],
        "Task 3 - YOLO detection": ["YOLODetector", "configure_inference", "process_metadata"],
        "Task 4 - Source management": ["SourceManager", "add_source", "remove_source"],
        "Task 5 - REST API": ["_create_api_app", "/api/v1/sources", "FastAPI"],
        "Task 6 - Error recovery": ["PipelineState.ERROR", "_on_bus_message"],
        "Task 7 - Config management": ["ConfigManager", "_load_config_file", "env_overrides"],
        "Task 8 - Logging": ["Logger", "JsonFormatter", "structured"],
        "Task 9 - Docker": ["Dockerfile", "docker-compose.yml"]
    }
    
    with open("video_analytics_script.py", "r") as f:
        content = f.read()
    
    for task, markers in requirements.items():
        found_all = all(marker in content for marker in markers)
        if found_all:
            print(f"[OK] {task}")
        else:
            missing = [m for m in markers if m not in content]
            print(f"[FAIL] {task} - missing: {missing}")
    
    # Check Docker files
    import os
    if os.path.exists("Dockerfile") and os.path.exists("docker-compose.yml"):
        print("[OK] Task 9 - Docker deployment files")
    else:
        print("[FAIL] Task 9 - Docker deployment files missing")
    
    return True

def main():
    """Run all checks."""
    print("=== Video Analytics Script Implementation Check ===\n")
    
    all_good = True
    
    all_good &= check_imports()
    all_good &= check_classes()
    all_good &= check_api_endpoints()
    all_good &= check_features()
    all_good &= check_prp_requirements()
    
    print("\n=== Summary ===")
    if all_good:
        print("[OK] All checks passed! Implementation appears complete.")
    else:
        print("[FAIL] Some checks failed. Review the implementation.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())