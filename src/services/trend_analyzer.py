"""
TrendAnalyzer — EMA-smoothed performance trend detection.

No LLM, no async, no external dependencies.
Pure math on a list of float scores.

Why EMA over simple moving average:
  Raw LLM scores are noisy. A candidate scoring [6.5, 8.0, 5.5, 7.5]
  has avg=6.875 — identical average to [7.5, 5.5, 8.0, 6.5] (reverse).
  EMA sees direction. SMA is blind to it.

α = 0.3: moderate smoothing.
  Lower → smoother but slow to react.
  Higher → reactive but noisy.
  0.3 responds to genuine 4-turn trends, ignores single outliers.
"""


class TrendAnalyzer:
    """
    EMA smoothed trend detection to preventy difficulty oscillation.

    Thresholds (both conditions required for adjustment):
     - Increase: avg_ema >=7.5 AND trend=="improving"
     - Decrease: avg_ema < 5.0 AND trend == "declining"

    Minimum 4 scores before any adjustment - insufficient data -> "stable".

    Fallback Protection:
        - Caller (Supervisor) is responsible for filtering is_fallback=True
        score BEFORE passing trajectory to this class.
        This class operates only on real scores.
    """
    MIN_TRAJECTORY_LENGTH = 4
    EMA_WINDOW = 4
    INCREASE_THRESHOLD = 7.5
    HIGH_STABLE_THRESHOLD = 8.0  # stable but excellent → still increase difficulty
    DECREASE_THRESHOLD = 5.0
    TREND_CHANGE_THRESHOLD = 0.8
    NEUTRAL_EMA = 5.0

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha < 1.0:
            raise ValueError(
                f"alpha must be between 0 and 1 (exclusive), got {alpha}. "
                f"Lower values = smoother, higher = more reactive."
            )
        self.alpha = alpha

    def calculate_ema(self, trajectory: list[float]) -> list[float]:
        """Calculate EMA for a full score trajectory.

        Args:
            trajectory (list[float]): Raw score trajectory

        Returns:
            list[float]: Smoothed EMA values, same length as input.
            Empty list if trajectory is empty.
        """
        if not trajectory:
            return []
        
        ema = [trajectory[0]]

        for score in trajectory[1:]:
            smoothed = self.alpha * score + (1-self.alpha)*ema[-1]
            ema.append(smoothed)

        return ema
    
    def get_trend(self, trajectory: list[float]) -> str:
        """
        Classify trajectory direction as "improving", "declining", or "stable".

        Operates on EMA of the FULL trajectory, then looks at the last
        EMA_WINDOW values. This is important:
          - EMA of last 4 raw scores != last 4 values of full EMA
          - Full EMA preserves historical context in each smoothed value
          - Slicing last 4 from full EMA gives noise-resistant recent window

        Returns "stable" if fewer than MIN_TRAJECTORY_LENGTH scores —
        not enough signal to confirm a genuine trend.

        Args:
            trajectory (list[float]): Raw score history

        Returns:
            str: One of: "improving" | "declining" | "stable"
        """
        if len(trajectory) < self.MIN_TRAJECTORY_LENGTH:
            return "stable"
        
        ema = self.calculate_ema(trajectory)
        recent_ema = ema[-self.EMA_WINDOW:]

        change = recent_ema[-1] - recent_ema[0]

        if change > self.TREND_CHANGE_THRESHOLD:
            return "improving"
        elif change < -self.TREND_CHANGE_THRESHOLD:
            return "declining"
        return "stable"
    
    def should_adjust_difficulty(
            self, trajectory: list[float]
    ) -> tuple[bool, str]:
        """
        Main decision function — called by Supervisor each turn after fan-in.

        Increase conditions (either is sufficient):
          1. trend == "improving" AND avg_ema >= INCREASE_THRESHOLD (7.5)
          2. trend == "stable"    AND avg_ema >= HIGH_STABLE_THRESHOLD (8.0)
             Rationale: a candidate already at 8.0+ EMA cannot produce an
             "improving" trend (EMA ceiling effect) but still deserves harder
             questions.
        Decrease condition:
          trend == "declining" AND avg_ema < DECREASE_THRESHOLD (5.0)

        Why require both trend + level for decrease?
          - Low EMA but improving → candidate is recovering, don't push down further.

        All possible return values:
          (False, "insufficient_data")  → fewer than 4 scores
          (True,  "increase")           → improving+high OR stable+excellent
          (True,  "decrease")           → declining + avg_ema < 5.0
          (False, "stable")             → all other cases

        Args:
            trajectory (list[float]): Raw score history

        Returns:
            tuple[bool, str]: (should_adjust: bool, direction: str)
        """
        if len(trajectory) < self.MIN_TRAJECTORY_LENGTH:
            return False, "insufficient_data"
        
        ema = self.calculate_ema(trajectory)
        recent_ema = ema[-self.EMA_WINDOW:]
        avg_ema = sum(recent_ema) / len(recent_ema)
        trend = self.get_trend(trajectory)

        if trend == "improving" and avg_ema >= self.INCREASE_THRESHOLD:
            return True, "increase"
        # Candidate is consistently excellent but can't show an "improving" trend
        # because EMA is already near the ceiling — still warrant harder questions.
        if trend == "stable" and avg_ema >= self.HIGH_STABLE_THRESHOLD:
            return True, "increase"
        if trend == "declining" and avg_ema < self.DECREASE_THRESHOLD:
            return True, "decrease"

        return False, "stable"
    
    def get_current_ema(self, trajectory: list[float]) -> float:
        """
        Convenience method — returns the latest EMA value.
        Used by Supervisor._observe() to populate current_ema in Observation.

        Returns NEUTRAL_EMA (5.0) for empty trajectory — the neutral
        midpoint of the 0-10 scale. Matches Supervisor's default:
            current_ema = ema[-1] if ema else 5.0

        Args:
            trajectory (list[float]): Raw score history

        Returns:
            float: Latest smoothed EMA value, or 5.0 if no data yet.
        """
        if not trajectory:
            return self.NEUTRAL_EMA

        ema = self.calculate_ema(trajectory)

        return ema[-1]