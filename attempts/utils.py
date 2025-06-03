from vllm import LLM, SamplingParams
import numpy as np
import torch
from collections import namedtuple
from vllm.model_executor.layers.logits_processor import _apply_logits_processors

ModelOutput = namedtuple("ModelOutput", ["logits"])

class CallableLLM:
    def __init__(self, *args, **kwargs):
        self.logits_list = []
        self.model = LLM(swap_space=0.5, *args, **kwargs) # !!! Must set swap_space=0.5, otherwise T4 will crash !!!
        self.hook_registered = False

    def _forward_hook(self, module, input, output):
        lm_head, hidden_states, sampling_metadata, *embedding_bias = input
        embedding_bias = embedding_bias[0] if embedding_bias else None
        logits = module._get_logits(hidden_states, lm_head, embedding_bias)
        if logits is not None:
            if module.soft_cap is not None:
                logits = logits / module.soft_cap
                logits = torch.tanh(logits)
                logits = logits * module.soft_cap
            if module.scale != 1.0:
                logits *= module.scale
            logits = _apply_logits_processors(logits, sampling_metadata)
            self.logits_list.append(logits.detach().cpu().numpy())
        return output

    def register_hook(self):
        if not self.hook_registered:
            model = self.model.llm_engine.model_executor.driver_worker.model_runner.model
            # model = self.model.llm_engine.model_executor.driver_worker.scorer_worker.model_runner.model
            model.logits_processor.register_forward_hook(self._forward_hook)
            self.hook_registered = True

    def __call__(self, batch):
        torch.cuda.empty_cache()
        self.register_hook()
        self.logits_list = []

        batch = batch.clone().tolist()
        sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1)
        self.model.generate(prompt_token_ids=batch, sampling_params=sampling_params, use_tqdm=False)
        logits = np.array([logit for logit in self.logits_list])

        return ModelOutput(logits=torch.from_numpy(logits).to('cuda:0'))

    def eval(self):
        pass

    def __getattr__(self, attr):
        return getattr(self.model, attr)