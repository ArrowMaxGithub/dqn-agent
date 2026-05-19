class EpsilonDecay:
    def __init__(
        self,
        initial_epsilon: float,
        final_epsilon: float,
        decay: float,
        schedule: str,
    ):
        self.initial = initial_epsilon
        self.final = final_epsilon
        self.decay = decay
        self.schedule = schedule

    def from_params(params):
        return EpsilonDecay(
            initial_epsilon=params["initial_epsilon"],
            final_epsilon=params["final_epsilon"],
            decay=params["iterations"] * params["epsilon_decay"],
            schedule=params["epsilon_schedule"],
        )

    def get(self, iter) -> float:
        match self.schedule:
            case "linear":
                return max(
                    self.final,
                    self.initial - (self.initial - self.final) * (iter / self.decay),
                )
            case _:
                raise ValueError(f"Unknown schedule: {self.schedule}")
