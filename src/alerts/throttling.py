"""
Advanced alert throttling with multiple algorithms and burst detection.

This module provides sophisticated throttling mechanisms including token bucket,
sliding window, and adaptive throttling to prevent alert spam while maintaining
system responsiveness.
"""

import asyncio
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import threading
from datetime import datetime, timedelta
import hashlib

from ..config import AppConfig, AlertConfig, AlertLevel
from ..utils.errors import AlertError, handle_error
from ..utils.logging import get_logger, performance_context
from ..utils.async_utils import get_task_manager, PeriodicTaskRunner
from ..detection.models import VideoDetection


class ThrottlingAlgorithm(Enum):
    """Available throttling algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class ThrottlingDecision(Enum):
    """Throttling decision outcomes."""
    ALLOW = "allow"
    THROTTLE = "throttle"
    BURST_LIMIT = "burst_limit"
    RATE_LIMIT = "rate_limit"
    COOLDOWN = "cooldown"


@dataclass
class ThrottlingState:
    """State information for throttling algorithms."""
    algorithm: ThrottlingAlgorithm
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    burst_count: int = 0
    burst_start_time: float = 0.0
    cooldown_until: float = 0.0
    total_requests: int = 0
    throttled_requests: int = 0
    last_allowed_time: float = 0.0
    consecutive_throttled: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'algorithm': self.algorithm.value,
            'tokens': self.tokens,
            'last_refill': self.last_refill,
            'requests': list(self.requests),
            'burst_count': self.burst_count,
            'burst_start_time': self.burst_start_time,
            'cooldown_until': self.cooldown_until,
            'total_requests': self.total_requests,
            'throttled_requests': self.throttled_requests,
            'last_allowed_time': self.last_allowed_time,
            'consecutive_throttled': self.consecutive_throttled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], algorithm: ThrottlingAlgorithm) -> 'ThrottlingState':
        """Create state from dictionary."""
        state = cls(algorithm=algorithm)
        state.tokens = data.get('tokens', 0.0)
        state.last_refill = data.get('last_refill', time.time())
        state.requests = deque(data.get('requests', []), maxlen=1000)
        state.burst_count = data.get('burst_count', 0)
        state.burst_start_time = data.get('burst_start_time', 0.0)
        state.cooldown_until = data.get('cooldown_until', 0.0)
        state.total_requests = data.get('total_requests', 0)
        state.throttled_requests = data.get('throttled_requests', 0)
        state.last_allowed_time = data.get('last_allowed_time', 0.0)
        state.consecutive_throttled = data.get('consecutive_throttled', 0)
        return state


@dataclass
class ThrottlingConfig:
    """Configuration for throttling algorithms."""
    # Token bucket parameters
    bucket_capacity: int = 10              # Maximum tokens
    refill_rate: float = 1.0              # Tokens per second
    
    # Sliding window parameters
    window_size_seconds: int = 60          # Window duration
    max_requests_per_window: int = 10      # Max requests in window
    
    # Burst detection
    burst_threshold: int = 5               # Requests to trigger burst detection
    burst_window_seconds: float = 5.0      # Burst detection window
    burst_cooldown_seconds: int = 30       # Cooldown after burst
    
    # Rate limiting
    min_interval_seconds: float = 1.0      # Minimum interval between requests
    max_requests_per_minute: int = 60      # Maximum requests per minute
    
    # Adaptive parameters
    adaptive_enabled: bool = True          # Enable adaptive throttling
    load_threshold: float = 0.8           # System load threshold
    backoff_multiplier: float = 2.0      # Backoff multiplier
    recovery_factor: float = 0.9          # Recovery factor
    
    # Priority handling
    priority_multipliers: Dict[str, float] = field(default_factory=lambda: {
        AlertLevel.LOW.value: 0.5,
        AlertLevel.MEDIUM.value: 1.0,
        AlertLevel.HIGH.value: 2.0,
        AlertLevel.CRITICAL.value: 5.0
    })


class TokenBucketThrottler:
    """Token bucket algorithm implementation."""
    
    def __init__(self, config: ThrottlingConfig, key: str):
        """
        Initialize token bucket throttler.
        
        Args:
            config: Throttling configuration
            key: Unique key for this throttler
        """
        self.config = config
        self.key = key
        self.logger = get_logger(__name__)
        self.state = ThrottlingState(ThrottlingAlgorithm.TOKEN_BUCKET)
        self.state.tokens = config.bucket_capacity
        self._lock = threading.Lock()
    
    def should_allow(self, priority_multiplier: float = 1.0) -> Tuple[ThrottlingDecision, Dict[str, Any]]:
        """
        Check if request should be allowed using token bucket algorithm.
        
        Args:
            priority_multiplier: Priority multiplier for token cost
            
        Returns:
            Tuple of (decision, metadata)
        """
        with self._lock:
            current_time = time.time()
            
            # Refill tokens based on elapsed time
            self._refill_tokens(current_time)
            
            # Calculate token cost (inverse of priority)
            token_cost = 1.0 / max(priority_multiplier, 0.1)
            
            # Check if we have enough tokens
            if self.state.tokens >= token_cost:
                # Allow request and consume tokens
                self.state.tokens -= token_cost
                self.state.total_requests += 1
                self.state.last_allowed_time = current_time
                self.state.consecutive_throttled = 0
                
                return ThrottlingDecision.ALLOW, {
                    'tokens_remaining': self.state.tokens,
                    'token_cost': token_cost,
                    'refill_rate': self.config.refill_rate
                }
            else:
                # Throttle request
                self.state.total_requests += 1
                self.state.throttled_requests += 1
                self.state.consecutive_throttled += 1
                
                # Calculate time until next token
                time_to_next_token = (token_cost - self.state.tokens) / self.config.refill_rate
                
                return ThrottlingDecision.THROTTLE, {
                    'tokens_remaining': self.state.tokens,
                    'token_cost': token_cost,
                    'time_to_next_token': time_to_next_token,
                    'consecutive_throttled': self.state.consecutive_throttled
                }
    
    def _refill_tokens(self, current_time: float):
        """Refill tokens based on elapsed time."""
        elapsed = current_time - self.state.last_refill
        tokens_to_add = elapsed * self.config.refill_rate
        
        self.state.tokens = min(
            self.config.bucket_capacity,
            self.state.tokens + tokens_to_add
        )
        self.state.last_refill = current_time
    
    def get_state(self) -> ThrottlingState:
        """Get current throttling state."""
        with self._lock:
            return self.state


class SlidingWindowThrottler:
    """Sliding window algorithm implementation."""
    
    def __init__(self, config: ThrottlingConfig, key: str):
        """
        Initialize sliding window throttler.
        
        Args:
            config: Throttling configuration
            key: Unique key for this throttler
        """
        self.config = config
        self.key = key
        self.logger = get_logger(__name__)
        self.state = ThrottlingState(ThrottlingAlgorithm.SLIDING_WINDOW)
        self._lock = threading.Lock()
    
    def should_allow(self, priority_multiplier: float = 1.0) -> Tuple[ThrottlingDecision, Dict[str, Any]]:
        """
        Check if request should be allowed using sliding window algorithm.
        
        Args:
            priority_multiplier: Priority multiplier for request limit
            
        Returns:
            Tuple of (decision, metadata)
        """
        with self._lock:
            current_time = time.time()
            
            # Clean old requests outside the window
            self._clean_old_requests(current_time)
            
            # Calculate effective limit based on priority
            effective_limit = int(self.config.max_requests_per_window * priority_multiplier)
            
            # Check if we're within the limit
            if len(self.state.requests) < effective_limit:
                # Allow request
                self.state.requests.append(current_time)
                self.state.total_requests += 1
                self.state.last_allowed_time = current_time
                self.state.consecutive_throttled = 0
                
                return ThrottlingDecision.ALLOW, {
                    'requests_in_window': len(self.state.requests),
                    'window_limit': effective_limit,
                    'window_size': self.config.window_size_seconds
                }
            else:
                # Throttle request
                self.state.total_requests += 1
                self.state.throttled_requests += 1
                self.state.consecutive_throttled += 1
                
                # Calculate time until window slides
                oldest_request = self.state.requests[0] if self.state.requests else current_time
                time_to_next_slot = (oldest_request + self.config.window_size_seconds) - current_time
                
                return ThrottlingDecision.RATE_LIMIT, {
                    'requests_in_window': len(self.state.requests),
                    'window_limit': effective_limit,
                    'time_to_next_slot': max(0, time_to_next_slot),
                    'consecutive_throttled': self.state.consecutive_throttled
                }
    
    def _clean_old_requests(self, current_time: float):
        """Remove requests outside the sliding window."""
        cutoff_time = current_time - self.config.window_size_seconds
        
        while self.state.requests and self.state.requests[0] < cutoff_time:
            self.state.requests.popleft()
    
    def get_state(self) -> ThrottlingState:
        """Get current throttling state."""
        with self._lock:
            return self.state


class BurstDetector:
    """Detects and handles burst patterns in alerts."""
    
    def __init__(self, config: ThrottlingConfig, key: str):
        """
        Initialize burst detector.
        
        Args:
            config: Throttling configuration
            key: Unique key for this detector
        """
        self.config = config
        self.key = key
        self.logger = get_logger(__name__)
        self._recent_requests: deque = deque(maxlen=100)
        self._burst_active = False
        self._burst_start_time = 0.0
        self._lock = threading.Lock()
    
    def detect_burst(self, current_time: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect if current request pattern indicates a burst.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            Tuple of (is_burst, metadata)
        """
        with self._lock:
            # Add current request
            self._recent_requests.append(current_time)
            
            # Clean old requests
            cutoff_time = current_time - self.config.burst_window_seconds
            while self._recent_requests and self._recent_requests[0] < cutoff_time:
                self._recent_requests.popleft()
            
            # Check for burst pattern
            requests_in_window = len(self._recent_requests)
            
            if requests_in_window >= self.config.burst_threshold:
                if not self._burst_active:
                    # New burst detected
                    self._burst_active = True
                    self._burst_start_time = current_time
                    self.logger.warning(
                        f"Burst detected for {self.key}: {requests_in_window} requests "
                        f"in {self.config.burst_window_seconds}s"
                    )
                
                return True, {
                    'burst_active': True,
                    'requests_in_burst_window': requests_in_window,
                    'burst_threshold': self.config.burst_threshold,
                    'burst_duration': current_time - self._burst_start_time
                }
            
            else:
                # Check if burst has ended
                if self._burst_active:
                    burst_duration = current_time - self._burst_start_time
                    if burst_duration > self.config.burst_window_seconds:
                        self._burst_active = False
                        self.logger.info(f"Burst ended for {self.key} after {burst_duration:.1f}s")
                
                return False, {
                    'burst_active': False,
                    'requests_in_burst_window': requests_in_window,
                    'burst_threshold': self.config.burst_threshold
                }
    
    def is_in_cooldown(self, current_time: float) -> bool:
        """Check if currently in post-burst cooldown."""
        if not self._burst_active and self._burst_start_time > 0:
            cooldown_end = self._burst_start_time + self.config.burst_cooldown_seconds
            return current_time < cooldown_end
        return False
    
    def get_cooldown_remaining(self, current_time: float) -> float:
        """Get remaining cooldown time in seconds."""
        if self.is_in_cooldown(current_time):
            cooldown_end = self._burst_start_time + self.config.burst_cooldown_seconds
            return max(0, cooldown_end - current_time)
        return 0.0


class AdaptiveThrottler:
    """Adaptive throttling that adjusts based on system load."""
    
    def __init__(self, config: ThrottlingConfig, key: str):
        """
        Initialize adaptive throttler.
        
        Args:
            config: Throttling configuration
            key: Unique key for this throttler
        """
        self.config = config
        self.key = key
        self.logger = get_logger(__name__)
        self._current_multiplier = 1.0
        self._system_load = 0.0
        self._load_history: deque = deque(maxlen=60)  # 1 minute of history
        self._last_adjustment = time.time()
        self._lock = threading.Lock()
    
    def update_system_load(self, load: float):
        """Update system load information."""
        with self._lock:
            self._system_load = load
            self._load_history.append((time.time(), load))
            
            # Adjust throttling multiplier based on load
            if load > self.config.load_threshold:
                # Increase throttling under high load
                self._current_multiplier *= self.config.backoff_multiplier
                self._current_multiplier = min(self._current_multiplier, 10.0)  # Cap at 10x
            elif load < self.config.load_threshold * 0.5:
                # Reduce throttling under low load
                self._current_multiplier *= self.config.recovery_factor
                self._current_multiplier = max(self._current_multiplier, 0.1)  # Floor at 0.1x
    
    def get_adjusted_config(self, base_config: ThrottlingConfig) -> ThrottlingConfig:
        """Get configuration adjusted for current system load."""
        with self._lock:
            adjusted_config = ThrottlingConfig(
                bucket_capacity=max(1, int(base_config.bucket_capacity / self._current_multiplier)),
                refill_rate=base_config.refill_rate / self._current_multiplier,
                window_size_seconds=base_config.window_size_seconds,
                max_requests_per_window=max(1, int(base_config.max_requests_per_window / self._current_multiplier)),
                burst_threshold=max(1, int(base_config.burst_threshold / self._current_multiplier)),
                burst_window_seconds=base_config.burst_window_seconds,
                burst_cooldown_seconds=int(base_config.burst_cooldown_seconds * self._current_multiplier),
                min_interval_seconds=base_config.min_interval_seconds * self._current_multiplier,
                max_requests_per_minute=max(1, int(base_config.max_requests_per_minute / self._current_multiplier))
            )
            
            return adjusted_config
    
    def get_load_info(self) -> Dict[str, Any]:
        """Get current load information."""
        with self._lock:
            return {
                'current_load': self._system_load,
                'load_threshold': self.config.load_threshold,
                'current_multiplier': self._current_multiplier,
                'load_history_size': len(self._load_history)
            }


class ThrottlingManager:
    """
    Comprehensive throttling manager with multiple algorithms and smart detection.
    
    Provides intelligent alert throttling using token bucket, sliding window,
    burst detection, and adaptive algorithms to prevent spam while maintaining
    system responsiveness.
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize throttling manager.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.alert_config = config.alerts
        self.logger = get_logger(__name__)
        
        # Throttling configuration
        self.throttling_config = ThrottlingConfig(
            bucket_capacity=self.alert_config.max_alerts_per_minute,
            refill_rate=self.alert_config.max_alerts_per_minute / 60.0,  # Per second
            window_size_seconds=60,
            max_requests_per_window=self.alert_config.max_alerts_per_minute,
            burst_threshold=self.alert_config.burst_threshold,
            burst_window_seconds=10.0,
            burst_cooldown_seconds=self.alert_config.throttle_seconds,
            min_interval_seconds=1.0,
            max_requests_per_minute=self.alert_config.max_alerts_per_minute
        )
        
        # Throttler instances
        self._global_throttler = TokenBucketThrottler(self.throttling_config, "global")
        self._source_throttlers: Dict[str, TokenBucketThrottler] = {}
        self._pattern_throttlers: Dict[str, SlidingWindowThrottler] = {}
        self._source_pattern_throttlers: Dict[str, TokenBucketThrottler] = {}
        
        # Burst detection
        self._global_burst_detector = BurstDetector(self.throttling_config, "global")
        self._source_burst_detectors: Dict[str, BurstDetector] = {}
        
        # Adaptive throttling
        self._adaptive_throttler = AdaptiveThrottler(self.throttling_config, "global")
        
        # State management
        self._state_file = Path("data/throttling_state.json")
        self._state_save_interval = 30.0  # Save state every 30 seconds
        self._periodic_runner = PeriodicTaskRunner(self._state_save_interval)
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'allowed_requests': 0,
            'throttled_requests': 0,
            'burst_detections': 0,
            'cooldown_blocks': 0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load saved state
        self._load_state()
        
        self.logger.info("ThrottlingManager initialized")
    
    async def should_allow_alert(
        self, 
        detection: VideoDetection,
        alert_level: AlertLevel = AlertLevel.MEDIUM
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if an alert should be allowed based on throttling rules.
        
        Args:
            detection: Detection that triggered the alert
            alert_level: Alert priority level
            
        Returns:
            Tuple of (should_allow, reason, metadata)
        """
        with self._lock:
            current_time = time.time()
            source_id = detection.source_id
            pattern_name = detection.pattern_name
            
            # Update statistics
            self._stats['total_requests'] += 1
            
            # Get priority multiplier
            priority_multiplier = self.throttling_config.priority_multipliers.get(
                alert_level.value, 1.0
            )
            
            # Check global burst detection
            is_burst, burst_metadata = self._global_burst_detector.detect_burst(current_time)
            
            if is_burst:
                self._stats['burst_detections'] += 1
                
                # Apply stricter throttling during burst
                priority_multiplier *= 0.1  # Reduce priority during burst
                
                self.logger.debug(f"Burst detected globally: {burst_metadata}")
            
            # Check global cooldown
            if self._global_burst_detector.is_in_cooldown(current_time):
                cooldown_remaining = self._global_burst_detector.get_cooldown_remaining(current_time)
                self._stats['cooldown_blocks'] += 1
                
                return False, "global_cooldown", {
                    'cooldown_remaining': cooldown_remaining,
                    'burst_metadata': burst_metadata
                }
            
            # Check global throttling
            global_decision, global_metadata = self._global_throttler.should_allow(priority_multiplier)
            
            if global_decision != ThrottlingDecision.ALLOW:
                self._stats['throttled_requests'] += 1
                return False, f"global_{global_decision.value}", global_metadata
            
            # Check source-specific throttling
            source_decision, source_metadata = await self._check_source_throttling(
                source_id, priority_multiplier, current_time
            )
            
            if source_decision != ThrottlingDecision.ALLOW:
                self._stats['throttled_requests'] += 1
                return False, f"source_{source_decision.value}", source_metadata
            
            # Check pattern-specific throttling
            pattern_decision, pattern_metadata = await self._check_pattern_throttling(
                pattern_name, priority_multiplier, current_time
            )
            
            if pattern_decision != ThrottlingDecision.ALLOW:
                self._stats['throttled_requests'] += 1
                return False, f"pattern_{pattern_decision.value}", pattern_metadata
            
            # Check source+pattern combination throttling
            combo_key = f"{source_id}:{pattern_name}"
            combo_decision, combo_metadata = await self._check_combination_throttling(
                combo_key, priority_multiplier, current_time
            )
            
            if combo_decision != ThrottlingDecision.ALLOW:
                self._stats['throttled_requests'] += 1
                return False, f"combo_{combo_decision.value}", combo_metadata
            
            # All checks passed - allow the alert
            self._stats['allowed_requests'] += 1
            
            return True, "allowed", {
                'priority_multiplier': priority_multiplier,
                'global_metadata': global_metadata,
                'source_metadata': source_metadata,
                'pattern_metadata': pattern_metadata,
                'combo_metadata': combo_metadata,
                'is_burst': is_burst
            }
    
    async def _check_source_throttling(
        self, 
        source_id: str, 
        priority_multiplier: float,
        current_time: float
    ) -> Tuple[ThrottlingDecision, Dict[str, Any]]:
        """Check source-specific throttling."""
        # Get or create source throttler
        if source_id not in self._source_throttlers:
            self._source_throttlers[source_id] = TokenBucketThrottler(
                self.throttling_config, f"source_{source_id}"
            )
        
        throttler = self._source_throttlers[source_id]
        
        # Check source-specific burst detection
        if source_id not in self._source_burst_detectors:
            self._source_burst_detectors[source_id] = BurstDetector(
                self.throttling_config, f"source_{source_id}"
            )
        
        burst_detector = self._source_burst_detectors[source_id]
        is_burst, burst_metadata = burst_detector.detect_burst(current_time)
        
        if is_burst:
            priority_multiplier *= 0.2  # Reduce priority for source burst
        
        # Check cooldown
        if burst_detector.is_in_cooldown(current_time):
            cooldown_remaining = burst_detector.get_cooldown_remaining(current_time)
            return ThrottlingDecision.COOLDOWN, {
                'source_id': source_id,
                'cooldown_remaining': cooldown_remaining
            }
        
        # Apply throttling
        decision, metadata = throttler.should_allow(priority_multiplier)
        metadata['source_id'] = source_id
        metadata['is_burst'] = is_burst
        
        return decision, metadata
    
    async def _check_pattern_throttling(
        self, 
        pattern_name: str, 
        priority_multiplier: float,
        current_time: float
    ) -> Tuple[ThrottlingDecision, Dict[str, Any]]:
        """Check pattern-specific throttling."""
        # Get or create pattern throttler
        if pattern_name not in self._pattern_throttlers:
            self._pattern_throttlers[pattern_name] = SlidingWindowThrottler(
                self.throttling_config, f"pattern_{pattern_name}"
            )
        
        throttler = self._pattern_throttlers[pattern_name]
        decision, metadata = throttler.should_allow(priority_multiplier)
        metadata['pattern_name'] = pattern_name
        
        return decision, metadata
    
    async def _check_combination_throttling(
        self, 
        combo_key: str, 
        priority_multiplier: float,
        current_time: float
    ) -> Tuple[ThrottlingDecision, Dict[str, Any]]:
        """Check source+pattern combination throttling."""
        # Get or create combination throttler
        if combo_key not in self._source_pattern_throttlers:
            # Use more restrictive config for combinations
            combo_config = ThrottlingConfig(
                bucket_capacity=max(1, self.throttling_config.bucket_capacity // 2),
                refill_rate=self.throttling_config.refill_rate / 2,
                window_size_seconds=self.throttling_config.window_size_seconds,
                max_requests_per_window=max(1, self.throttling_config.max_requests_per_window // 2)
            )
            
            self._source_pattern_throttlers[combo_key] = TokenBucketThrottler(
                combo_config, f"combo_{combo_key}"
            )
        
        throttler = self._source_pattern_throttlers[combo_key]
        decision, metadata = throttler.should_allow(priority_multiplier)
        metadata['combo_key'] = combo_key
        
        return decision, metadata
    
    def update_system_load(self, load: float):
        """Update system load for adaptive throttling."""
        self._adaptive_throttler.update_system_load(load)
        
        # Update throttling configuration based on load
        if self.throttling_config.adaptive_enabled:
            adjusted_config = self._adaptive_throttler.get_adjusted_config(self.throttling_config)
            
            # Update global throttler with adjusted config
            self._global_throttler.config = adjusted_config
    
    def get_throttling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive throttling statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Add throttler statistics
            stats['global_state'] = self._global_throttler.get_state().to_dict()
            stats['source_throttlers'] = len(self._source_throttlers)
            stats['pattern_throttlers'] = len(self._pattern_throttlers)
            stats['combo_throttlers'] = len(self._source_pattern_throttlers)
            
            # Add adaptive throttling info
            stats['adaptive_info'] = self._adaptive_throttler.get_load_info()
            
            # Calculate rates
            if stats['total_requests'] > 0:
                stats['allow_rate'] = stats['allowed_requests'] / stats['total_requests']
                stats['throttle_rate'] = stats['throttled_requests'] / stats['total_requests']
            else:
                stats['allow_rate'] = 0.0
                stats['throttle_rate'] = 0.0
            
            return stats
    
    def get_source_statistics(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific source."""
        with self._lock:
            if source_id in self._source_throttlers:
                throttler = self._source_throttlers[source_id]
                return {
                    'source_id': source_id,
                    'state': throttler.get_state().to_dict(),
                    'has_burst_detector': source_id in self._source_burst_detectors
                }
            return None
    
    def reset_throttling(self, scope: str = "all"):
        """
        Reset throttling state.
        
        Args:
            scope: Scope to reset ("all", "global", "sources", "patterns", "combos")
        """
        with self._lock:
            if scope in ["all", "global"]:
                self._global_throttler = TokenBucketThrottler(self.throttling_config, "global")
                self._global_burst_detector = BurstDetector(self.throttling_config, "global")
            
            if scope in ["all", "sources"]:
                self._source_throttlers.clear()
                self._source_burst_detectors.clear()
            
            if scope in ["all", "patterns"]:
                self._pattern_throttlers.clear()
            
            if scope in ["all", "combos"]:
                self._source_pattern_throttlers.clear()
            
            if scope == "all":
                self._stats = {
                    'total_requests': 0,
                    'allowed_requests': 0,
                    'throttled_requests': 0,
                    'burst_detections': 0,
                    'cooldown_blocks': 0
                }
            
            self.logger.info(f"Reset throttling state for scope: {scope}")
    
    def _save_state(self):
        """Save throttling state to file."""
        try:
            state_data = {
                'timestamp': time.time(),
                'stats': self._stats,
                'global_throttler': self._global_throttler.get_state().to_dict(),
                'source_throttlers': {
                    source_id: throttler.get_state().to_dict()
                    for source_id, throttler in self._source_throttlers.items()
                },
                'pattern_throttlers': {
                    pattern: throttler.get_state().to_dict()
                    for pattern, throttler in self._pattern_throttlers.items()
                },
                'combo_throttlers': {
                    combo: throttler.get_state().to_dict()
                    for combo, throttler in self._source_pattern_throttlers.items()
                }
            }
            
            # Ensure data directory exists
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write state file
            with open(self._state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.debug("Saved throttling state")
        
        except Exception as e:
            self.logger.error(f"Error saving throttling state: {e}")
    
    def _load_state(self):
        """Load throttling state from file."""
        try:
            if not self._state_file.exists():
                self.logger.debug("No throttling state file found, starting fresh")
                return
            
            with open(self._state_file, 'r') as f:
                state_data = json.load(f)
            
            # Load statistics
            self._stats.update(state_data.get('stats', {}))
            
            # Load global throttler state
            if 'global_throttler' in state_data:
                self._global_throttler.state = ThrottlingState.from_dict(
                    state_data['global_throttler'],
                    ThrottlingAlgorithm.TOKEN_BUCKET
                )
            
            self.logger.info("Loaded throttling state from file")
        
        except Exception as e:
            self.logger.error(f"Error loading throttling state: {e}")
    
    async def start_periodic_tasks(self):
        """Start periodic maintenance tasks."""
        await self._periodic_runner.start_periodic_task(
            "save_state",
            self._save_state_task
        )
        
        self.logger.info("Started throttling periodic tasks")
    
    async def stop_periodic_tasks(self):
        """Stop periodic maintenance tasks."""
        await self._periodic_runner.stop_all_tasks()
        
        # Save final state
        self._save_state()
        
        self.logger.info("Stopped throttling periodic tasks")
    
    async def _save_state_task(self):
        """Periodic state saving task."""
        self._save_state()


# Global throttling manager instance
_global_throttling_manager: Optional[ThrottlingManager] = None


def get_throttling_manager(config: Optional[AppConfig] = None) -> Optional[ThrottlingManager]:
    """Get global throttling manager instance."""
    global _global_throttling_manager
    if _global_throttling_manager is None and config is not None:
        _global_throttling_manager = ThrottlingManager(config)
    return _global_throttling_manager