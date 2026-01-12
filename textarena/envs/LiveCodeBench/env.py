from datasets import load_dataset
from typing import Tuple
import textarena as ta
import numpy as np
import json
from pathlib import Path
import multiprocessing
import hashlib

from textarena.envs.LiveCodeBench.utils.compute_code_generation_metrics import _temp_run


class LiveCodeBenchEnv(ta.Env):
    def __init__(self, version_tag: str = 'release_v5', timeout: int = 30, debug: bool = False):
        super().__init__()
        self.version_tag = version_tag
        self.timeout = timeout
        self.debug = debug

    def reset(self, num_players, seed = None):
        self.state = ta.SinglePlayerState(num_players=1, seed=seed, max_turns=1)
        dataset = load_dataset("json", data_files={"test": "/home/tg69/TextArena/textarena/envs/LiveCodeBench/data/livecodebench_v5.jsonl"}, split="test")
        self.state.reset(game_state=dataset[seed % len(dataset)], player_prompt_function=lambda player_id, game_state: game_state['prompt'])
    
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        sample = self.state.game_state
        code, tests = self._extract_code(action), sample['tests']
        res = self.check_correctness(tests=tests, generation=code, timeout=self.timeout, debug=self.debug)
        assert res["md5"] == tests["md5"], "test md5 mismatched"
        if res["ispass"]: self.state.set_outcome(reward=1.0, reason="Correct solution.")
        else: self.state.set_outcome(reward=-1.0, reason="Incorrect solution.")
        self.state.game_info[self.state.current_player_id] = {**res["metadata"], **self.state.game_info[self.state.current_player_id]}
        return self.state.step()
    
    def download_data(self):
        from textarena.envs.LiveCodeBench.utils.process_data import download_data
        return download_data()
    
    def _extract_code(self, text: str) -> str:
        outputlines = text.split("\n")
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
        if len(indexlines) < 2: return ""
        return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

    def check_correctness(self, tests: dict, generation: str, timeout: int = 30, debug: bool = False):
        tests_path = Path("/home/tg69/TextArena/textarena/envs/LiveCodeBench/data/livecodebench_v5_tests") / tests["fname"]
        with open(tests_path, "r") as f: sample = json.load(f)
        md5 = self.calculate_string_md5(json.dumps(sample))
        manager = multiprocessing.Manager()
        result = manager.list()
        metadata_list = manager.list()
        p = multiprocessing.Process(
            target=_temp_run,
            args=(sample, generation, debug, result, metadata_list, timeout),
        )
        p.start()
        p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)
        if p.is_alive():
            p.kill()
        if not result:
            in_outs = json.loads(sample["input_output"])
            result = [[-1 for i in range(len(in_outs["inputs"]))]]
            metadata_list = [{"error_code": -3}]
            if debug: print(f"global timeout")

        res, metadata = result[0], metadata_list[0]
        fixed = []
        for e in res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            if e != True and e != False:
                e = False
            fixed.append(e)
        res = fixed
        if not np.all(res):
            print("fail")
            return dict(ispass=0, md5=md5, results=res, metadata=metadata)
        else:
            print("pass")
            return dict(ispass=1, md5=md5, results=res, metadata=metadata)

    def calculate_string_md5(self,input_string: str):
        md5 = hashlib.md5()
        md5.update(input_string.encode("utf-8"))
        return md5.hexdigest()