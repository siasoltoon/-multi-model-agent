from .models import ModelEndpoint


class SmartRouter:
    """Select the healthiest endpoint using a transparent weighted score."""

    def choose(self, endpoints: list[ModelEndpoint]) -> ModelEndpoint:
        available = [e for e in endpoints if e.score() >= 0]
        if not available:
            raise RuntimeError("No healthy model endpoint with available quota")
        return max(available, key=ModelEndpoint.score)
