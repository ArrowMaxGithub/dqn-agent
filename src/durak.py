from dataclasses import dataclass, field
from enum import Enum
import random


# Farben (Suits)
class CardColor(Enum):
    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3


# Werte (Ranks)
class CardValue(Enum):
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14


# Einzelne Karte
@dataclass(frozen=True)
class Card:
    value: CardValue
    color: CardColor

    def __repr__(self):
        return f"{self.value.name} {self.color.name}"


# default_factory=list = „mach mir jedes Mal eine neue leere Liste“,
# sonst teilen sich alle dieselbe Liste


# Hand eines Spielers
@dataclass
class PlayerHand:
    cards: list[Card] = field(default_factory=list)


# Spieler
@dataclass
class Player:
    name: str
    player_num: int
    hand: PlayerHand = field(default_factory=PlayerHand)


# Angriff + Verteidigung als Paar
@dataclass
class TablePair:
    # Eine Angriffskarte liegt immer.
    attack: Card

    # Solange noch nicht verteidigt wurde, ist defense = None.
    defense: Card | None = None


class GameState:
    def __init__(self):
        # Rundenzähler, wird beim Nachziehen inkrementiert
        self.round = 0

        # Trumpffarbe; vor setup() noch unbekannt.
        self.trump: CardColor | None = None

        # Ziehstapel, offene Trumpfkarte draw_pile[0].
        self.draw_pile: list[Card] = []

        # Spielerzustände.
        self.players: list[Player] = []

        # Indizes in self.players.
        self.attacker: int = 0
        self.defender: int = 1

        # Karten auf dem Tisch.
        self.table: list[TablePair] = []

        # Karten, die schon aus dem Spiel sind.
        self.discard_pile: list[Card] = []

    # ----------------------------
    # SETUP
    # ----------------------------

    # Interne Methoden-Namen beginnen mit Unterstrich, Reihenfolge nicht in Methode nicht relevant.
    # Da wir Enum im Enum haben bei card:
    # card.value = CardValue.NAME z.B. King, card.value.value = INT z.B. 13.

    def setup(self, player_count: int) -> None:
        if player_count != 2:
            raise ValueError("This implementation only supports 2 players.")

        deck = self._create_shuffled_deck()

        self.players = [
            Player(name=f"Player {i + 1}", player_num=i) for i in range(player_count)
        ]

        # Abwechselnd je 6 Karten austeilen.
        for _ in range(6):
            for player in self.players:
                player.hand.cards.append(deck.pop())

        # Die unterste, offene Karte bestimmt den Trumpf; ablegen in Index 0.
        trump_card = deck.pop()
        self.trump = trump_card.color

        # dahinter der restliche Nachziehstapel.
        self.draw_pile = [trump_card] + deck

        # Startspieler: niedrigster Trumpf beginnt.
        self.attacker = self._find_first_attacker()
        self.defender = (self.attacker + 1) % player_count

        # Zu Beginn liegt noch nichts auf dem Tisch und der Ablagestapel ist leer.
        self.table = []
        self.discard_pile = []

    def set_player_name(self, index: int, name: str):
        if index < len(self.players):
            self.players[index].name = name

    def _create_shuffled_deck(self) -> list[Card]:
        deck: list[Card] = []
        for color in CardColor:
            for value in CardValue:
                deck.append(Card(value=value, color=color))
        random.shuffle(deck)
        return deck

    def _find_first_attacker(self) -> int:
        # Sucht pro Spieler den kleinsten Trumpf.
        best_player_index = 0
        best_trump_value: int | None = None

        for i, player in enumerate(self.players):
            for card in player.hand.cards:
                if card.color == self.trump:
                    if best_trump_value is None or card.value.value < best_trump_value:
                        best_trump_value = card.value.value
                        best_player_index = i

        # Falls niemand Trumpf hat, startet Spieler 0.
        return best_player_index

    # ----------------------------
    # HILFSFUNKTIONEN FÜR DEN TISCH
    # ----------------------------
    def table_values(self) -> set[CardValue]:
        values: set[CardValue] = set()
        for pair in self.table:
            values.add(pair.attack.value)
            if pair.defense is not None:
                values.add(pair.defense.value)
        return values

    def last_open_attack(self) -> Card | None:
        # Letzte noch nicht verteidigte Angriffskarte suchen.
        for pair in reversed(self.table):
            if pair.defense is None:
                return pair.attack
        return None

    def is_fully_defended(self) -> bool:
        return len(self.table) > 0 and all(
            pair.defense is not None for pair in self.table
        )

    def attack_count(self) -> int:
        # Anzahl gelegter Angriffskarten in dieser Runde.
        return len(self.table)

    def max_attack_count(self) -> int:
        # Maximal 6 Angriffe, aber wenn der Verteidiger weniger Karten hatte,
        # ist die Obergrenze seine Kartenanzahl zu Beginn der Runde.
        defender_cards = len(self.players[self.defender].hand.cards)
        return min(6, defender_cards)

    def can_add_more_attack_cards(self) -> bool:
        return self.attack_count() < self.max_attack_count()

    def add_attack_card(self, card: Card) -> None:
        self.table.append(TablePair(attack=card))

    def add_defense_card(self, card: Card) -> None:
        for pair in reversed(self.table):
            if pair.defense is None:
                pair.defense = card
                return
        raise ValueError("No open attack card to defend against.")

    def collect_table_cards(self) -> list[Card]:
        # Alle Tischkarten in einer Liste zurückgeben und Tisch leeren.
        cards: list[Card] = []
        for pair in self.table:
            cards.append(pair.attack)
            if pair.defense is not None:
                cards.append(pair.defense)
        self.table = []
        return cards

    def discard_table_cards(self) -> None:
        # Nur aufrufen, wenn alle Angriffe verteidigt wurden.
        for pair in self.table:
            self.discard_pile.append(pair.attack)
            if pair.defense is not None:
                self.discard_pile.append(pair.defense)
        self.table = []

    def refill_hands(self) -> None:
        # Nachziehen beginnt beim Angreifer, dann Verteidiger.
        order = [self.attacker, self.defender]
        for player_index in order:
            player = self.players[player_index]
            while len(player.hand.cards) < 6 and len(self.draw_pile) > 0:
                # Die "oberste" ziehbare Karte nehmen wir vom Ende.
                # draw_pile[0] bleibt damit bis zuletzt die offene Trumpfkarte.
                card = self.draw_pile.pop()
                player.hand.cards.append(card)

        # Wenn nur noch die offene Trumpfkarte übrig ist, kann sie normal gezogen werden.
        # Dieser Fall ist bereits durch pop() oben abgedeckt, sobald sie am Ende angekommen ist.

        self.round += 1

    def swap_roles(self) -> None:
        self.attacker, self.defender = self.defender, self.attacker

    # ----------------------------
    # REGELLOGIK: LEGALE KARTEN
    # ----------------------------
    def LegalAttackCards(self, cards: list[Card]) -> list[Card]:
        # Wenn dies der erste Angriff der Runde ist, darf jede Karte gespielt werden.
        if not self.table:
            return list(cards)

        # Weitere Angriffskarten müssen einen Wert haben, der bereits irgendwo auf dem Tisch liegt.
        values_on_table = self.table_values()
        return [card for card in cards if card.value in values_on_table]

    def LegalDefenseCards(self, cards: list[Card]) -> list[Card]:
        attack_card = self.last_open_attack()
        if attack_card is None:
            return []

        legal_cards: list[Card] = []

        for card in cards:
            # Gleiche Farbe und höherer Wert schlägt.
            if (
                card.color == attack_card.color
                and card.value.value > attack_card.value.value
            ):
                legal_cards.append(card)
                continue

            # Trumpf schlägt Nicht-Trumpf.
            if card.color == self.trump and attack_card.color != self.trump:
                legal_cards.append(card)
                continue

        return legal_cards

    # ----------------------------
    # SPIELENDE
    # ----------------------------
    def is_game_over(self) -> bool:
        # Spiel endet erst sinnvoll, wenn der Nachziehstapel leer ist
        # und mindestens ein Spieler keine Karten mehr hat.
        if len(self.draw_pile) > 0:
            return False

        return any(len(player.hand.cards) == 0 for player in self.players)

    def winner_index(self) -> int | None:
        if not self.is_game_over():
            return None

        # Wer keine Karten mehr hat, gewinnt.
        for i, player in enumerate(self.players):
            if len(player.hand.cards) == 0:
                return i
        return None
