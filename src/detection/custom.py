"""
Custom detection strategy registration and dynamic loading.

This module provides infrastructure for registering custom detection patterns,
dynamic loading of user-defined strategies, and plugin-style architecture
for extensible detection capabilities.
"""

import asyncio
import importlib
import inspect
import sys
from typing import Dict, List, Optional, Any, Type, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json
import yaml

from ..config import AppConfig
from ..utils.errors import DetectionError, handle_error
from ..utils.logging import get_logger, performance_context
from .models import DetectionStrategy, VideoDetection, DetectionResult, ModelInfo


@dataclass
class StrategyPlugin:
    """Information about a strategy plugin."""
    name: str
    version: str
    author: str
    description: str
    strategy_class: Type[DetectionStrategy]
    module_path: str
    config_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    supported_formats: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    enabled: bool = True


@dataclass
class CustomPattern:
    """Custom detection pattern definition."""
    name: str
    description: str
    pattern_type: str  # 'template', 'feature', 'model', 'custom'
    config: Dict[str, Any]
    created_by: str
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class StrategyRegistry:
    """Registry for managing custom detection strategies."""
    
    def __init__(self):
        """Initialize strategy registry."""
        self.logger = get_logger(__name__)
        self._plugins: Dict[str, StrategyPlugin] = {}
        self._patterns: Dict[str, CustomPattern] = {}
        self._strategy_instances: Dict[str, DetectionStrategy] = {}
        self._plugin_paths: List[Path] = []
        
        # Default plugin search paths
        self._default_plugin_paths = [
            Path("plugins/strategies"),
            Path("~/.pyds-app/plugins").expanduser(),
            Path("/usr/local/share/pyds-app/plugins")
        ]
    
    def add_plugin_path(self, path: Union[str, Path]):
        """Add a path to search for strategy plugins."""
        plugin_path = Path(path)
        if plugin_path not in self._plugin_paths:
            self._plugin_paths.append(plugin_path)
            self.logger.info(f"Added plugin search path: {plugin_path}")
    
    def register_strategy_plugin(
        self, 
        plugin_info: StrategyPlugin,
        force_reload: bool = False
    ) -> bool:
        """
        Register a strategy plugin.
        
        Args:
            plugin_info: Plugin information
            force_reload: Force reload if plugin already exists
            
        Returns:
            True if registration successful
        """
        try:
            plugin_name = plugin_info.name
            
            if plugin_name in self._plugins and not force_reload:
                self.logger.warning(f"Plugin {plugin_name} already registered")
                return False
            
            # Validate strategy class
            if not issubclass(plugin_info.strategy_class, DetectionStrategy):
                raise DetectionError(f"Plugin {plugin_name} strategy class must inherit from DetectionStrategy")
            
            # Validate required methods
            required_methods = ['initialize', 'detect', 'cleanup']
            for method_name in required_methods:
                if not hasattr(plugin_info.strategy_class, method_name):
                    raise DetectionError(f"Plugin {plugin_name} missing required method: {method_name}")
            
            # Store plugin
            self._plugins[plugin_name] = plugin_info
            
            self.logger.info(f"Registered strategy plugin: {plugin_name} v{plugin_info.version}")
            return True
        
        except Exception as e:
            error = handle_error(e, context={'plugin': plugin_info.name})
            self.logger.error(f"Failed to register plugin {plugin_info.name}: {error}")
            return False
    
    def unregister_strategy_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a strategy plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistration successful
        """
        try:
            if plugin_name not in self._plugins:
                self.logger.warning(f"Plugin {plugin_name} not found")
                return False
            
            # Clean up any active instances
            if plugin_name in self._strategy_instances:
                strategy = self._strategy_instances[plugin_name]
                asyncio.create_task(strategy.cleanup())
                del self._strategy_instances[plugin_name]
            
            # Remove plugin
            del self._plugins[plugin_name]
            
            self.logger.info(f"Unregistered strategy plugin: {plugin_name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error unregistering plugin {plugin_name}: {e}")
            return False
    
    def load_plugins_from_directory(self, plugin_dir: Path) -> int:
        """
        Load strategy plugins from directory.
        
        Args:
            plugin_dir: Directory containing plugin modules
            
        Returns:
            Number of plugins loaded
        """
        loaded_count = 0
        
        try:
            if not plugin_dir.exists():
                self.logger.debug(f"Plugin directory does not exist: {plugin_dir}")
                return 0
            
            self.logger.info(f"Loading plugins from: {plugin_dir}")
            
            # Add plugin directory to Python path
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            # Find Python files
            plugin_files = list(plugin_dir.glob("*.py"))
            plugin_files = [f for f in plugin_files if not f.name.startswith("__")]
            
            for plugin_file in plugin_files:
                try:
                    module_name = plugin_file.stem
                    
                    # Import module
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Look for plugin registration function or strategy classes
                    if hasattr(module, 'register_plugin'):
                        # Plugin has registration function
                        success = await self._call_plugin_register(module.register_plugin)
                        if success:
                            loaded_count += 1
                    
                    elif hasattr(module, 'get_plugin_info'):
                        # Plugin has info function
                        plugin_info = module.get_plugin_info()
                        if isinstance(plugin_info, StrategyPlugin):
                            if self.register_strategy_plugin(plugin_info):
                                loaded_count += 1
                    
                    else:
                        # Look for strategy classes directly
                        strategy_classes = self._find_strategy_classes(module)
                        for strategy_class in strategy_classes:
                            plugin_info = self._create_plugin_info_from_class(
                                strategy_class, plugin_file
                            )
                            if self.register_strategy_plugin(plugin_info):
                                loaded_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error loading plugin {plugin_file}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} plugins from {plugin_dir}")
            return loaded_count
        
        except Exception as e:
            self.logger.error(f"Error loading plugins from {plugin_dir}: {e}")
            return loaded_count
    
    async def _call_plugin_register(self, register_func: Callable) -> bool:
        """Call plugin registration function."""
        try:
            if asyncio.iscoroutinefunction(register_func):
                return await register_func(self)
            else:
                return register_func(self)
        except Exception as e:
            self.logger.error(f"Error calling plugin register function: {e}")
            return False
    
    def _find_strategy_classes(self, module) -> List[Type[DetectionStrategy]]:
        """Find DetectionStrategy classes in module."""
        strategy_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj != DetectionStrategy and 
                issubclass(obj, DetectionStrategy) and 
                obj.__module__ == module.__name__):
                strategy_classes.append(obj)
        
        return strategy_classes
    
    def _create_plugin_info_from_class(
        self, 
        strategy_class: Type[DetectionStrategy], 
        module_path: Path
    ) -> StrategyPlugin:
        """Create plugin info from strategy class."""
        # Extract metadata from class
        class_name = strategy_class.__name__
        doc = inspect.getdoc(strategy_class) or "Custom strategy plugin"
        
        return StrategyPlugin(
            name=class_name.lower().replace('strategy', '').replace('detection', ''),
            version="1.0",
            author="Unknown",
            description=doc,
            strategy_class=strategy_class,
            module_path=str(module_path)
        )
    
    async def create_strategy_instance(
        self, 
        plugin_name: str, 
        instance_name: str,
        config: Dict[str, Any]
    ) -> Optional[DetectionStrategy]:
        """
        Create instance of registered strategy.
        
        Args:
            plugin_name: Name of registered plugin
            instance_name: Name for strategy instance
            config: Strategy configuration
            
        Returns:
            Strategy instance or None if creation failed
        """
        try:
            if plugin_name not in self._plugins:
                raise DetectionError(f"Plugin {plugin_name} not registered")
            
            plugin_info = self._plugins[plugin_name]
            
            # Create strategy instance
            strategy = plugin_info.strategy_class(instance_name, config)
            
            # Initialize strategy
            if await strategy.initialize():
                self._strategy_instances[instance_name] = strategy
                self.logger.info(f"Created strategy instance: {instance_name} ({plugin_name})")
                return strategy
            else:
                self.logger.error(f"Failed to initialize strategy instance: {instance_name}")
                return None
        
        except Exception as e:
            error = handle_error(e, context={
                'plugin': plugin_name,
                'instance': instance_name
            })
            self.logger.error(f"Error creating strategy instance: {error}")
            return None
    
    def get_registered_plugins(self) -> Dict[str, StrategyPlugin]:
        """Get all registered plugins."""
        return self._plugins.copy()
    
    def get_plugin_info(self, plugin_name: str) -> Optional[StrategyPlugin]:
        """Get information about a specific plugin."""
        return self._plugins.get(plugin_name)
    
    def get_strategy_instance(self, instance_name: str) -> Optional[DetectionStrategy]:
        """Get strategy instance by name."""
        return self._strategy_instances.get(instance_name)
    
    def list_available_plugins(self) -> List[str]:
        """List names of available plugins."""
        return list(self._plugins.keys())
    
    def register_custom_pattern(self, pattern: CustomPattern) -> bool:
        """
        Register a custom detection pattern.
        
        Args:
            pattern: Custom pattern definition
            
        Returns:
            True if registration successful
        """
        try:
            if pattern.name in self._patterns:
                self.logger.warning(f"Pattern {pattern.name} already registered")
                return False
            
            # Validate pattern configuration
            if not self._validate_pattern_config(pattern):
                return False
            
            self._patterns[pattern.name] = pattern
            self.logger.info(f"Registered custom pattern: {pattern.name}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error registering pattern {pattern.name}: {e}")
            return False
    
    def _validate_pattern_config(self, pattern: CustomPattern) -> bool:
        """Validate custom pattern configuration."""
        try:
            # Basic validation
            if not pattern.name or not pattern.pattern_type:
                self.logger.error("Pattern must have name and type")
                return False
            
            # Type-specific validation
            if pattern.pattern_type == 'template':
                required_keys = ['template_path', 'threshold']
                if not all(key in pattern.config for key in required_keys):
                    self.logger.error(f"Template pattern missing required config: {required_keys}")
                    return False
            
            elif pattern.pattern_type == 'feature':
                required_keys = ['reference_images', 'detector_type']
                if not all(key in pattern.config for key in required_keys):
                    self.logger.error(f"Feature pattern missing required config: {required_keys}")
                    return False
            
            return True
        
        except Exception as e:
            self.logger.error(f"Error validating pattern config: {e}")
            return False
    
    def get_custom_patterns(self) -> Dict[str, CustomPattern]:
        """Get all registered custom patterns."""
        return self._patterns.copy()
    
    def get_custom_pattern(self, pattern_name: str) -> Optional[CustomPattern]:
        """Get custom pattern by name."""
        return self._patterns.get(pattern_name)
    
    async def load_all_plugins(self) -> int:
        """Load plugins from all configured paths."""
        total_loaded = 0
        
        # Load from default paths
        for plugin_path in self._default_plugin_paths:
            if plugin_path.exists():
                count = self.load_plugins_from_directory(plugin_path)
                total_loaded += count
        
        # Load from additional paths
        for plugin_path in self._plugin_paths:
            if plugin_path.exists():
                count = self.load_plugins_from_directory(plugin_path)
                total_loaded += count
        
        return total_loaded
    
    def save_patterns_to_file(self, file_path: Path) -> bool:
        """Save registered patterns to file."""
        try:
            patterns_data = {}
            for name, pattern in self._patterns.items():
                patterns_data[name] = {
                    'name': pattern.name,
                    'description': pattern.description,
                    'pattern_type': pattern.pattern_type,
                    'config': pattern.config,
                    'created_by': pattern.created_by,
                    'created_at': pattern.created_at.isoformat(),
                    'version': pattern.version,
                    'enabled': pattern.enabled,
                    'metadata': pattern.metadata
                }
            
            # Save as YAML for readability
            with open(file_path, 'w') as f:
                yaml.dump(patterns_data, f, default_flow_style=False)
            
            self.logger.info(f"Saved {len(patterns_data)} patterns to {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving patterns to {file_path}: {e}")
            return False
    
    def load_patterns_from_file(self, file_path: Path) -> int:
        """Load patterns from file."""
        loaded_count = 0
        
        try:
            if not file_path.exists():
                self.logger.debug(f"Patterns file does not exist: {file_path}")
                return 0
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() == '.json':
                    patterns_data = json.load(f)
                else:
                    patterns_data = yaml.safe_load(f)
            
            for pattern_data in patterns_data.values():
                try:
                    # Convert datetime string back to datetime object
                    created_at = datetime.fromisoformat(pattern_data['created_at'])
                    
                    pattern = CustomPattern(
                        name=pattern_data['name'],
                        description=pattern_data['description'],
                        pattern_type=pattern_data['pattern_type'],
                        config=pattern_data['config'],
                        created_by=pattern_data['created_by'],
                        created_at=created_at,
                        version=pattern_data.get('version', '1.0'),
                        enabled=pattern_data.get('enabled', True),
                        metadata=pattern_data.get('metadata', {})
                    )
                    
                    if self.register_custom_pattern(pattern):
                        loaded_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error loading pattern {pattern_data.get('name', 'unknown')}: {e}")
            
            self.logger.info(f"Loaded {loaded_count} patterns from {file_path}")
            return loaded_count
        
        except Exception as e:
            self.logger.error(f"Error loading patterns from {file_path}: {e}")
            return 0


class CustomStrategyLoader:
    """Loader for custom detection strategies with dependency management."""
    
    def __init__(self, registry: StrategyRegistry):
        """
        Initialize custom strategy loader.
        
        Args:
            registry: Strategy registry instance
        """
        self.registry = registry
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._dependency_cache: Dict[str, bool] = {}
    
    async def load_strategy_from_config(
        self, 
        config: Dict[str, Any],
        strategy_name: str
    ) -> Optional[DetectionStrategy]:
        """
        Load strategy from configuration.
        
        Args:
            config: Strategy configuration
            strategy_name: Name for strategy instance
            
        Returns:
            Loaded strategy instance or None
        """
        try:
            strategy_type = config.get('type', 'custom')
            plugin_name = config.get('plugin', strategy_type)
            
            # Check if plugin is registered
            if plugin_name not in self.registry.get_registered_plugins():
                self.logger.error(f"Plugin {plugin_name} not registered")
                return None
            
            # Check dependencies
            plugin_info = self.registry.get_plugin_info(plugin_name)
            if not await self._check_dependencies(plugin_info.dependencies):
                self.logger.error(f"Dependencies not satisfied for plugin {plugin_name}")
                return None
            
            # Create strategy instance
            strategy = await self.registry.create_strategy_instance(
                plugin_name, strategy_name, config
            )
            
            return strategy
        
        except Exception as e:
            error = handle_error(e, context={
                'strategy_name': strategy_name,
                'config': config
            })
            self.logger.error(f"Error loading strategy from config: {error}")
            return None
    
    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        for dependency in dependencies:
            if dependency not in self._dependency_cache:
                try:
                    # Try to import the dependency
                    importlib.import_module(dependency)
                    self._dependency_cache[dependency] = True
                except ImportError:
                    self._dependency_cache[dependency] = False
                    self.logger.warning(f"Dependency not available: {dependency}")
            
            if not self._dependency_cache[dependency]:
                return False
        
        return True
    
    def create_strategy_template(self, strategy_type: str) -> str:
        """Create a template for custom strategy implementation."""
        template = f'''"""
Custom {strategy_type} detection strategy.

This is a template for implementing custom detection strategies.
Replace this with your actual implementation.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

from pyds_app.detection.models import DetectionStrategy, VideoDetection, DetectionResult, BoundingBox, DetectionMetadata
from pyds_app.utils.logging import get_logger
from pyds_app.utils.errors import handle_error


class Custom{strategy_type.title()}Strategy(DetectionStrategy):
    """Custom {strategy_type} detection strategy."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize custom strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
        """
        super().__init__(name, config)
        self.logger = get_logger(f"{{__name__}}.{{self.__class__.__name__}}")
        
        # TODO: Initialize your strategy-specific components here
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the strategy."""
        try:
            self.logger.info(f"Initializing custom {strategy_type} strategy: {{self.name}}")
            
            # TODO: Add your initialization logic here
            # - Load models
            # - Initialize components
            # - Validate configuration
            
            self._initialized = True
            self.logger.info(f"Custom {strategy_type} strategy initialized successfully")
            return True
        
        except Exception as e:
            error = handle_error(e, context={{'strategy': self.name}})
            self.logger.error(f"Failed to initialize strategy: {{error}}")
            return False
    
    async def detect(self, frame_data: Any, source_id: str, frame_number: int) -> DetectionResult:
        """
        Perform detection on frame data.
        
        Args:
            frame_data: Frame data to process
            source_id: Source identifier
            frame_number: Frame number
            
        Returns:
            Detection results
        """
        start_time = time.perf_counter()
        detections = []
        
        try:
            if not self._initialized:
                raise RuntimeError("Strategy not initialized")
            
            # TODO: Implement your detection logic here
            # - Process frame_data
            # - Run detection algorithm
            # - Create VideoDetection objects
            
            # Example placeholder detection
            detection = VideoDetection(
                pattern_name="custom_object",
                confidence=0.8,
                bounding_box=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.3),
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                metadata=DetectionMetadata(
                    class_name="custom_object",
                    model_name=self.name
                )
            )
            detections.append(detection)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return DetectionResult(
                detections=detections,
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                processing_time_ms=processing_time
            )
        
        except Exception as e:
            error = handle_error(e, context={{
                'strategy': self.name,
                'source_id': source_id,
                'frame_number': frame_number
            }})
            self.logger.error(f"Detection error: {{error}}")
            
            return DetectionResult(
                detections=[],
                frame_number=frame_number,
                source_id=source_id,
                timestamp=datetime.now(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def should_process(self, detection: VideoDetection) -> bool:
        """
        Determine if this strategy should process the given detection.
        
        Args:
            detection: Detection to evaluate
            
        Returns:
            True if strategy should process the detection
        """
        # TODO: Implement your filtering logic here
        return self.enabled
    
    async def cleanup(self) -> None:
        """Clean up strategy resources."""
        try:
            self.logger.info(f"Cleaning up custom {strategy_type} strategy: {{self.name}}")
            
            # TODO: Add your cleanup logic here
            # - Release resources
            # - Close connections
            # - Save state if needed
            
            self._initialized = False
            self.logger.info("Custom {strategy_type} strategy cleanup completed")
        
        except Exception as e:
            self.logger.error(f"Error during cleanup: {{e}}")


# Plugin registration function
def register_plugin(registry):
    """Register the custom strategy plugin."""
    from pyds_app.detection.custom import StrategyPlugin
    
    plugin_info = StrategyPlugin(
        name="custom_{strategy_type}",
        version="1.0",
        author="Your Name",
        description="Custom {strategy_type} detection strategy",
        strategy_class=Custom{strategy_type.title()}Strategy,
        module_path=__file__,
        dependencies=[],  # Add any required dependencies here
        supported_formats=["numpy", "gstreamer"]  # Supported input formats
    )
    
    return registry.register_strategy_plugin(plugin_info)
'''
        return template


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_strategy_registry() -> StrategyRegistry:
    """Get global strategy registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


# Convenience functions for custom pattern management
def register_custom_pattern(
    name: str,
    pattern_type: str,
    config: Dict[str, Any],
    description: str = "",
    created_by: str = "user"
) -> bool:
    """
    Register a custom detection pattern.
    
    Args:
        name: Pattern name
        pattern_type: Type of pattern ('template', 'feature', 'model', 'custom')
        config: Pattern configuration
        description: Pattern description
        created_by: Creator identifier
        
    Returns:
        True if registration successful
    """
    registry = get_strategy_registry()
    
    pattern = CustomPattern(
        name=name,
        description=description,
        pattern_type=pattern_type,
        config=config,
        created_by=created_by
    )
    
    return registry.register_custom_pattern(pattern)


def load_custom_patterns_from_file(file_path: Union[str, Path]) -> int:
    """
    Load custom patterns from file.
    
    Args:
        file_path: Path to patterns file
        
    Returns:
        Number of patterns loaded
    """
    registry = get_strategy_registry()
    return registry.load_patterns_from_file(Path(file_path))


def save_custom_patterns_to_file(file_path: Union[str, Path]) -> bool:
    """
    Save custom patterns to file.
    
    Args:
        file_path: Path to save patterns
        
    Returns:
        True if save successful
    """
    registry = get_strategy_registry()
    return registry.save_patterns_to_file(Path(file_path))


async def load_strategy_plugins(plugin_directories: List[Union[str, Path]] = None) -> int:
    """
    Load strategy plugins from directories.
    
    Args:
        plugin_directories: List of directories to search for plugins
        
    Returns:
        Number of plugins loaded
    """
    registry = get_strategy_registry()
    
    if plugin_directories:
        for plugin_dir in plugin_directories:
            registry.add_plugin_path(Path(plugin_dir))
    
    return await registry.load_all_plugins()


def create_strategy_template_file(
    strategy_type: str, 
    output_path: Union[str, Path]
) -> bool:
    """
    Create a template file for custom strategy implementation.
    
    Args:
        strategy_type: Type of strategy template to create
        output_path: Path to save template file
        
    Returns:
        True if template created successfully
    """
    try:
        loader = CustomStrategyLoader(get_strategy_registry())
        template_content = loader.create_strategy_template(strategy_type)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(template_content)
        
        logger = get_logger(__name__)
        logger.info(f"Created strategy template: {output_file}")
        return True
    
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating strategy template: {e}")
        return False