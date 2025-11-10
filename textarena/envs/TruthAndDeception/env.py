import json, os, re, random
import importlib.resources
from typing import Optional, Tuple, Dict, Any

import textarena as ta 
from textarena.envs.TruthAndDeception.renderer import create_board_str

class TruthAndDeceptionEnv(ta.Env):
    def __init__(self, max_turns: Optional[int]=5, data_path: Optional[str]=None):
        assert max_turns%2==0, f"Please use an even number of max turns. Current max_turns: {max_turns}"
        self.max_turns = max_turns
        self._load_facts(data_path=data_path)
        self.guess_fact1_pattern = re.compile(r"\[Final Guess: Fact 1\]", re.IGNORECASE)
        self.guess_fact2_pattern = re.compile(r"\[Final Guess: Fact 2\]", re.IGNORECASE)

    def get_board_str(self):
        return create_board_str(game_state=self.state.game_state)

    def _load_facts(self, data_path: Optional[str]) -> None:
        try:
            if data_path is not None: # Use provided path
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"Facts data file not found at: {data_path}")
                with open(data_path, "r", encoding="utf-8") as file:
                    self.facts_data = json.load(file)
            else: # Use package resource
                with importlib.resources.files('textarena.envs.TruthAndDeception').joinpath('facts.json').open('r') as file:
                    self.facts_data = json.load(file)
        except Exception as e:
            raise FileNotFoundError(f"Failed to load facts data: {str(e)}")

    def reset(self, num_players: int, seed: Optional[int]=None):
        self.state = ta.TwoPlayerState(num_players=num_players, max_turns=self.max_turns, seed=seed)
        selected_facts = random.choice(self.facts_data) # Select a random set of facts
        facts = [ # Randomize the order in which the facts are presented
            (selected_facts["facts"]["fact1"], selected_facts["correct_fact"]=="fact1"),
            (selected_facts["facts"]["fact2"], selected_facts["correct_fact"]=="fact2"),
        ]
        random.shuffle(facts)
        game_state={
                "fact1": {"fact": facts[0][0], "is_correct": facts[0][1]}, "fact2": {"fact": facts[1][0], "is_correct": facts[1][1]},
                "correct_fact": facts[0][0] if facts[0][1] else facts[1][0], "wrong_fact": facts[0][0] if facts[1][1] else facts[1][0]
            }
        self.state.reset(game_state=game_state, player_prompt_function=self._prompt, role_mapping={0: "Deceiver", 1: "Guesser"})

    def _prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        role = self.state.role_mapping[player_id]
        prompt = f"You are Player {player_id}, and you are the {role} in the 'Truth and Deception' game.\n"
        prompt += (
            "When discussing with the other player, think through your strategy privately before you speak. When you are ready to send the public message, format it as [Message: <your words>].\n"
            "Only the content inside [Message: ...] is delivered to the other player; anything outside the brackets is kept hidden.\n"
            "When you make the final guess, send it as [Final Guess: Fact 1] or [Final Guess: Fact 2].\n"
        )
        if role == "Deceiver":
            prompt += (
                f"You know which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']} ({'correct' if game_state['fact1']['is_correct'] else 'wrong'})\n"
                f"Fact 2: {game_state['fact2']['fact']} ({'correct' if game_state['fact2']['is_correct'] else 'wrong'})\n"
                "Your goal is to convince the other player to choose the incorrect fact.\n"
                f"You will have a conversation for {self.state.max_turns} turns.\n"
                "After that, the other player will make their guess."
            )
        elif role == "Guesser":
            prompt += (
                f"You need to determine which of the following two facts is correct:\n"
                f"Fact 1: {game_state['fact1']['fact']}\n"
                f"Fact 2: {game_state['fact2']['fact']}\n"
                f"You will have a conversation with the other player for {self.state.max_turns} turns.\n"
                "After that, you will make your guess."
            )
        else: raise ValueError(f"Unexpected role mapping: {role}. Expected 'Deceiver' or 'Guesser'.")
        return prompt 

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        if not self.state.turn == self.state.max_turns-1: broadcast_message = self._extract_broadcast_message(action=action) 
        else: broadcast_message = action
        self.state.add_observation(from_id=self.state.current_player_id, message=broadcast_message, observation_type=ta.ObservationType.PLAYER_ACTION)
        if self.state.turn == self.state.max_turns-2:  # check if the guessing phase has started
            self.state.add_observation(message="Now guess which of the two facts is correct. Share any final reasoning privately, then send '[Final Guess: Fact 1]' or '[Final Guess: Fact 2]'.", observation_type=ta.ObservationType.GAME_MESSAGE) 
        elif self.state.turn == self.state.max_turns-1:
            if self.guess_fact1_pattern.search(broadcast_message) or self.guess_fact2_pattern.search(broadcast_message): # evaluate guess
                if (self.guess_fact1_pattern.search(broadcast_message) and self.state.game_state["fact1"]["is_correct"]) or (self.guess_fact2_pattern.search(broadcast_message) and self.state.game_state["fact2"]["is_correct"]):
                    winner_id=self.state.current_player_id; reason=f"Player {self.state.current_player_id} guessed correct fact." # correct guess
                else: winner_id=1-self.state.current_player_id; reason=f"Player {self.state.current_player_id} guessed the wrong fact." # wrong guess
                self.state.set_winner(player_id=winner_id, reason=reason) # set state winner
            else: self.state.set_invalid_move(reason=f"Player {self.state.current_player_id} did not make their guess in the correct format.")
        return self.state.step()

    def _extract_broadcast_message(self, action: str) -> str:
        match = re.search(r"\[Message:\s*(.*?)\]", action, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else action.strip()
