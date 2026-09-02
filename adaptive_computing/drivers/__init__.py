"""
AC drivers. These modules provide templates for common tasks
of using the samplers, surrogates, datasets, and evaluators
together in a goal-driven task.
"""

from adaptive_computing.drivers.active_base import ActiveLoopDriver
from adaptive_computing.drivers.active_cost_ratio import ActiveLoopDriverCostRatio
from adaptive_computing.drivers.query_validators import get_query_validator

# Optional Hero driver import
try:
    from adaptive_computing.drivers.hero import ActiveLoopDriverHero
except ImportError:
    # Hero not available - this is expected in basic AC environment
    pass

try:
    from adaptive_computing.drivers.hero_cost_ratio import ActiveLoopDriverHeroCostRatio
except ImportError:
    pass
