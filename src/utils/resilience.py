"""
Resilience patterns (Circuit Breaker).
Addresses Gap 9: No Circuit Breaker.
"""

import time
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CircuitBreakerOpen(Exception):
    pass

class SimpleCircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def allow_request(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
                logger.info("[CircuitBreaker] Transitioning to HALF-OPEN")
                return True
            return False
        return True

    def record_success(self):
        if self.state == "HALF-OPEN":
            self.state = "CLOSED"
            self.failures = 0
            logger.info("[CircuitBreaker] Transitioning to CLOSED (Recovered)")
        else:
            self.failures = 0

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"[CircuitBreaker] Circuit OPENED after {self.failures} failures")

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not self.allow_request():
                raise CircuitBreakerOpen(f"Circuit is OPEN. Retrying in {int(self.recovery_timeout - (time.time() - self.last_failure_time))}s")
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise e
        return wrapper
