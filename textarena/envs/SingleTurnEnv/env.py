from datasets import load_dataset
from typing import Tuple
import textarena as ta
import re


DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class SingleTurnDatasetEnv(ta.Env):
    
    def __init__(self, dataset: str = 'scripts/SingleTurnEnv/data/aime2025.jsonl', verifier: str = 'math-verify'):
        super().__init__()
        self.verifier = verifier
        if verifier == 'judge': self.judge = ta.agents.OpenRouterAgent(model_name='openai/gpt-4o-2024-11-20')
        self.dataset = load_dataset("json", data_files=dataset)['train']

    def reset(self, num_players, seed = None):
        self.state = ta.SinglePlayerState(num_players=1, seed=seed, max_turns=1)
        self.state.reset(game_state=self.dataset[seed % len(self.dataset)], player_prompt_function=self._generate_player_prompt)
    
    def _generate_player_prompt(self, player_id: int, game_state: dict) -> str: return str(game_state["question"])
    
    def llm_as_a_judge(self, solution: str, response: str) -> bool: 
        prompt = DEFAULT_JUDGE_PROMPT.format(answer=solution, response=response)
        match = re.search(r"(yes|no)", self.judge(prompt))
        result = match.group(1) if match else "no"
        return result == "yes"
    
    def math_verify(self, solution: str, response: str) -> bool:
        from math_verify import verify, parse
        solution = parse(solution); response = parse(response)
        return verify(solution, response)
        
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        match = re.compile(r".*\[(.*?)\]").search(action)
        if match:
            action = match.group(1)
            if self.verifier == 'math-verify': verify = self.math_verify
            else: verify = self.llm_as_a_judge
            if verify(self.state.game_state['answer'], action): self.state.set_outcome(reward=1.0, reason="Correct solution.")
            else: self.state.set_outcome(reward=-1.0, reason="Incorrect solution.")
        else: self.state.set_invalid_move(reward=-1.0, reason=f"Invalid format. The player did not respond with a valid answer in square brackets.")
        return self.state.step()
