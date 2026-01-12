import random
from datasets import load_dataset
from typing import List, Optional, Tuple
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


# Per-subset regex patterns for answer extraction in BBH
BBH_ANSWER_PATTERNS = {
    # Yes/No subsets
    "causal_judgement": r"\b(Yes|No|yes|no)\b",
    "navigate": r"\b(Yes|No|yes|no)\b",
    "sports_understanding": r"\b(Yes|No|yes|no)\b",
    "web_of_lies": r"\b(Yes|No|yes|no)\b",
    
    # True/False subsets
    "boolean_expressions": r"\b(True|False|true|false)\b",
    
    # Valid/Invalid subsets
    "formal_fallacies": r"\b(valid|invalid|Valid|Invalid)\b",
    
    # Multiple choice (A)/(B)/(C) etc.
    "disambiguation_qa": r"\(([A-C])\)",
    "hyperbaton": r"\(([A-B])\)",
    "logical_deduction_five_objects": r"\(([A-E])\)",
    "logical_deduction_seven_objects": r"\(([A-G])\)",
    "logical_deduction_three_objects": r"\(([A-C])\)",
    "movie_recommendation": r"\(([A-E])\)",
    "penguins_in_a_table": r"\(([A-E])\)",
    "ruin_names": r"\(([A-D])\)",
    "salient_translation_error_detection": r"\(([A-F])\)",
    "snarks": r"\(([A-B])\)",
    "temporal_sequences": r"\(([A-D])\)",
    "reasoning_about_colored_objects": r"\(([A-R])\)",
    "tracking_shuffled_objects_five_objects": r"\(([A-E])\)",
    "tracking_shuffled_objects_seven_objects": r"\(([A-G])\)",
    "tracking_shuffled_objects_three_objects": r"\(([A-C])\)",
    
    # Free-form text answers (no specific pattern, use full response)
    "date_understanding": None,
    "dyck_languages": None,
    "geometric_shapes": None,
    "multistep_arithmetic_two": None,
    "object_counting": None,
    "word_sorting": None,
}


class BigBenchHardEnv(ta.Env):
    """
    Single-turn BBH task where the agent answers one question drawn from the dataset.
    Uses per-subset regex patterns for answer extraction, with fallback to string-match.
    If subset is None, iterates over all examples; otherwise filters to specified subset.
    """

    def __init__(self, subset: Optional[str] = None, verifier: str = "string-match"):
        super().__init__()
        self.verifier = verifier
        self.subset = subset
        if verifier == 'judge':
            self.judge = ta.agents.OpenRouterAgent(model_name='openai/gpt-4o-2024-11-20')
        
        # Load from local JSONL, optionally filter by subset
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data", "bbh.jsonl")
        full_dataset = load_dataset("json", data_files=data_path)["train"]
        if subset is not None:
            self.dataset = full_dataset.filter(lambda x: x["subset"] == subset)
        else:
            self.dataset = full_dataset

    def __len__(self):
        return len(self.dataset)

    def reset(self, num_players, seed=None):
        self.state = ta.SinglePlayerState(num_players=1, seed=seed, max_turns=1)
        self.state.reset(game_state=self.dataset[seed % len(self.dataset)], player_prompt_function=self._generate_player_prompt)

    def _generate_player_prompt(self, player_id: int, game_state: dict) -> str:
        prompt = (
            "Please reason step by step, and put your final answer within \\boxed{}.\n\n"
            f"{game_state['input']}"
        )
        return prompt

    def _normalize(self, text: str) -> str:
        """Standard BBH normalization: lowercase and collapse whitespace"""
        return re.sub(r"\s+", " ", text).strip().lower()

    def string_match(self, targets: List[str], response: str) -> bool:
        """Check if normalized response matches any normalized target"""
        normalized_response = self._normalize(response)
        for target in targets:
            if self._normalize(target) == normalized_response:
                return True
        return False

    def llm_as_a_judge(self, targets: List[str], response: str) -> bool:
        # Fall back to first target as the ground truth for judge comparison
        prompt = DEFAULT_JUDGE_PROMPT.format(answer=targets[0], response=response)
        result = self.judge(prompt).lower()
        match = re.search(r"\b(yes|no)\b", result)
        return match.group(1) == "yes" if match else False

    def _extract_answer(self, action: str, subset: str) -> Optional[str]:
        """Extract answer from \\boxed{} content, then apply subset-specific normalization."""
        # Extract content from \boxed{...}
        boxed_match = re.search(r'\\boxed\{([^}]*)\}', action)
        if not boxed_match:
            return None
        
        boxed_content = boxed_match.group(1).strip()
        
        # Apply subset-specific pattern to normalize the boxed content
        answer_pattern = BBH_ANSWER_PATTERNS.get(subset)
        if answer_pattern:
            match = re.search(answer_pattern, boxed_content)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        
        # No pattern or no match - return the boxed content as-is
        return boxed_content

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        # Get the subset for this sample to determine extraction pattern
        sample_subset = self.state.game_state.get("subset", "")
        extracted_answer = self._extract_answer(action, sample_subset)
        
        if extracted_answer is None:
            self.state.set_invalid_move(
                reward=-1.0,
                reason="Invalid format. Expected answer in \\boxed{}."
            )
            return self.state.step()
        
        # Get targets - handle both 'target' and 'targets' field names
        targets = self.state.game_state.get("target") or self.state.game_state.get("targets", [])
        if not isinstance(targets, list):
            targets = [targets]
        
        # Verify answer
        if self.verifier == 'judge':
            is_correct = self.llm_as_a_judge(targets, extracted_answer)
        else:  # default to string-match
            is_correct = self.string_match(targets, extracted_answer)
        
        if is_correct:
            self.state.set_outcome(reward=1.0, reason="Correct solution.")
        else:
            self.state.set_outcome(reward=-1.0, reason=f"Incorrect. Expected one of: {targets}")
        
        return self.state.step()
