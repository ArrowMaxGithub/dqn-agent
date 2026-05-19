import numpy as np

from durak import Action, Card, CardColor, CardValue
from durak_env import Status


class PlayerInterface:
    def OnTurn(
        self,
        attacking_card: Card | None,
        hand_cards: list[Card],
        legal_cards: list[Card],
    ) -> Action: ...

    def GetName(self) -> str: ...


class AgentInterface:
    def GetAction(self, obs_dict: dict, force_exploitation=False) -> int: ...

    def GetName(self) -> str: ...


# Adapter for agents to be used in a standard Durak game
# Keeps book about all known cards for the agent
class AgentAdpater(PlayerInterface):
    def __init__(self, trump: CardColor, agent: AgentInterface):
        self.num_cards = len(CardColor) * len(CardValue)
        self.card_states = np.concatenate(
            (np.array([Status.Unknown] * self.num_cards), trump)
        ).astype(np.int8)
        self.passing_action = self.num_cards
        self.agent = agent
        self.was_attacking_last_turn = False

    def OnTurn(
        self,
        attacking_card: Card | None,
        hand_cards: list[Card],
        legal_cards: list[Card],
    ) -> Action:
        expected_num_hand_cards = np.count_nonzero(self.card_states == Status.MyCard)

        # Must play card on first move
        has_drawn_cards = len(hand_cards) != expected_num_hand_cards
        is_attacking = attacking_card is None
        has_changed_roles = is_attacking != self.was_attacking_last_turn
        is_new_turn = has_drawn_cards or has_changed_roles
        can_pass = (not is_attacking) or (not is_new_turn)

        if is_new_turn:
            mask = np.isin(self.card_states, [Status.Attack, Status.Defense])
            self.card_states[mask] = (
                Status.OpponentCard
                if self.was_attacking_last_turn  # Opponent must have taken the cards in play since the agent is still attacking
                else Status.Discarded  # Otherwise the cards were discarded after a successfull defense
            )

        # Set opponent's attacking card
        if attacking_card:
            index = self._get_index_from_card(attacking_card)
            self.card_states[index] = Status.Attack

        # Set agent hand cards
        indices = [self._get_index_from_card(card) for card in hand_cards]
        self.card_states[indices] = Status.MyCard

        observations = np.copy(self.card_states)

        # Generate action mask of legal moves
        # Start will all-zeros
        action_mask = np.zeros(self.num_cards + 1, dtype=np.int8)
        # Passing is allowed if the agent is defending or it it not the first (attack) turn
        action_mask[self.passing_action] = can_pass
        # Set ones for legal cards
        indices = [self._get_index_from_card(card) for card in legal_cards]
        action_mask[indices] = 1

        # Generate observations dict for agent
        obs_dict = {
            "observations": observations,
            "action_mask": action_mask,
        }

        # Return action from agent
        action = self.agent.GetAction(obs_dict=obs_dict, force_exploitation=True)
        card = self._get_card_from_index(action)

        if card:
            # Agent played a card as attacker or defender
            self.card_states[action] = Status.Attack if is_attacking else Status.Defense
        else:
            # Agent passed as attacker => Successful defense => Cards will be discarded
            # Agent passed as defender => Unsuccessful defense => Cards will be taken
            mask = np.isin(self.card_states, [Status.Attack, Status.Defense])
            self.card_states[mask] = Status.Discarded if is_attacking else Status.MyCard

        self.was_attacking_last_turn = is_attacking

        return card

    def GetName(self) -> str:
        return self.agent.GetName()

    def _get_index_from_card(self, card: Card) -> int:
        # [Spades[6..Ace], Clubs[6..Ace], Hearts[6..Ace], Diamonds[6..Ace]]
        return (card.color.value) * len(CardValue) + (card.value.value - 6)

    def _get_card_from_index(self, index: int) -> Card | None:
        if index == self.passing_action:
            return None

        color = CardColor(index // len(CardValue))
        value = CardValue(index % len(CardValue) + 6)
        return Card(value=value, color=color)
