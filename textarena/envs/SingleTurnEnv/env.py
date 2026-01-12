import random
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

    def __init__(
        self,
        dataset: str = 'scripts/SingleTurnEnv/data/aime2025.jsonl',
        verifier: str = 'math-verify'
    ):
        super().__init__()
        self.verifier = verifier

        if verifier == 'judge':
            self.judge = ta.agents.OpenRouterAgent(
                model_name='openai/gpt-4o-2024-11-20'
            )

        self.dataset = load_dataset(
            "json",
            data_files=dataset
        )['train']

    def reset(self, num_players, seed=None):
        self.state = ta.SinglePlayerState(
            num_players=1,
            seed=seed,
            max_turns=1
        )

        raw_game_state = self.dataset[seed % len(self.dataset)]

        # ---------- MULTIPLE CHOICE ----------
        if self.verifier == 'multiple-choice':
            options = raw_game_state["options"]
            correct_index = raw_game_state["correct_index"]

            indexed_options = list(enumerate(options))
            random.shuffle(indexed_options)

            new_correct_index = next(
                i for i, (orig_i, _) in enumerate(indexed_options)
                if orig_i == correct_index
            )

            option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            game_state = {
                "question": raw_game_state["question"],
                "options": [opt for _, opt in indexed_options],
                "answer": option_letters[new_correct_index],
            }

        # ---------- NON-MULTIPLE CHOICE ----------
        else:
            game_state = raw_game_state

        self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )

    def _generate_player_prompt(self, player_id: int, game_state: dict) -> str:
        if self.verifier == 'multiple-choice':
            option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            prompt = (
                "Please reason step by step, and put your final answer within "
                "\\boxed{}.\n"
                "Your final answer should be of the form: \\boxed{LETTER}.\n\n"
            )

            prompt += f"Question: {game_state['question']}\n\n"
            prompt += "Choices:\n"

            for i, option in enumerate(game_state["options"]):
                prompt += f"{option_letters[i]}) {option}\n"

            return prompt

        return str(game_state["question"])

    # ---------- VERIFIERS ----------

    def llm_as_a_judge(self, solution: str, response: str) -> bool:
        prompt = DEFAULT_JUDGE_PROMPT.format(
            answer=solution,
            response=response
        )
        match = re.search(r"(yes|no)", self.judge(prompt))
        result = match.group(1) if match else "no"
        return result == "yes"

    def math_verify(self, solution: str, response: str) -> bool:
        from math_verify import verify, parse
        solution = parse(solution)
        response = parse(response)
        return verify(solution, response)

    def exact_match(self, solution: str, response: str) -> bool:
        return solution.strip() == response.strip()

    # ---------- STEP ----------

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        match = re.search(r'\[(.*)\]', action)

        if match:
            action = match.group(1)

            if self.verifier == 'multiple-choice':
                verify = self.exact_match
            elif self.verifier == 'math-verify':
                verify = self.math_verify
            elif self.verifier == 'judge':
                verify = self.llm_as_a_judge
            else:
                verify = self.exact_match

            if verify(self.state.game_state["answer"], action):
                self.state.set_outcome(
                    reward=1.0,
                    reason="Correct solution."
                )
            else:
                self.state.set_outcome(
                    reward=-1.0,
                    reason="Incorrect solution."
                )
        else:
            self.state.set_invalid_move(
                reward=-1.0,
                reason="Invalid format. Expected answer in \\boxed{}."
            )

        return self.state.step()
