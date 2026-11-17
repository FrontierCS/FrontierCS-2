"""Read-only public rollout rendering and parsing contract; no verifier hook.

Candidates may import RolloutContract and parse_response from this module.
The verifier uses its independent, trusted copy.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import re

REASONS = frozenset({'action_provenance', 'prompt_provenance', 'assistant_provenance'})
SPECIAL_TOKENS = ('<think>', '</think>', '<answer>', '</answer>', '<|im_start|>', '<|im_end|>')


class ProvenanceError(RuntimeError):
    def __init__(self, reason):
        if reason not in REASONS:
            raise ValueError('unknown provenance reason')
        self.reason = reason
        super().__init__(reason)


def parse_response(raw, *, enable_think, action_sep, max_actions):
    """Pinned reference normalization, including its malformed-output behavior."""
    pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if enable_think else r'<answer>(.*?)</answer>'
    match = re.search(pattern, raw, re.DOTALL)
    if match is None:
        return raw, []
    thought, answer = (match.group(1), match.group(2)) if enable_think else ('', match.group(1))
    for token in SPECIAL_TOKENS:
        answer = answer.replace(token, '').strip()
        thought = thought.replace(token, '').strip()
    actions = [action.strip() for action in answer.split(action_sep) if action.strip()]
    if len(actions) > max_actions:
        actions = actions[:max_actions]
        answer = (' ' + action_sep + ' ').join(actions)
    normalized = f'<answer>{answer}</answer>'
    if enable_think:
        normalized = f'<think>{thought}</think>' + normalized
    return normalized, actions


def _list(value):
    return value.tolist() if hasattr(value, 'tolist') else list(value)


class RolloutContract:
    def __init__(self, config, tokenizer, registry, mode):
        self.config = copy.deepcopy(config)
        self.tokenizer = tokenizer
        self.enable_think = bool(config.agent_proxy.enable_think)
        self.action_sep = str(config.agent_proxy.action_sep)
        self.max_actions = int(config.agent_proxy.max_actions_per_turn)
        self.prefixes, self.max_tokens, self.vocab = {}, {}, {}
        cfg = config.es_manager[mode]
        # Reproduce fixed original instructions independently of ContextManager.
        current_group = 0
        for tag, groups in zip(cfg.env_configs.tags, cfg.env_configs.n_groups):
            custom = config.custom_envs[tag]
            env = asdict(registry[custom.env_type]())
            env.update(dict(custom.items()))
            prefix = env.get('env_instruction', '')
            if env.get('grid_vocab'):
                prefix += '\nThe meaning of each symbol in the state is:\n' + ', '.join(f'{k}: {v}' for k, v in env['grid_vocab'].items())
            if env.get('action_lookup'):
                prefix += '\nYour available actions are:\n' + ', '.join(str(v) for v in env['action_lookup'].values())
                prefix += f'\nYou can make up to {env["max_actions_per_traj"]} actions, separated by the action separator " {self.action_sep} "\n'
            vocab = {str(v).lower(): k for k, v in env['action_lookup'].items()}
            for env_id in range(current_group * cfg.group_size, (current_group + groups) * cfg.group_size):
                self.prefixes[env_id] = prefix
                self.max_tokens[env_id] = custom.get('max_tokens', config.actor_rollout_ref.rollout.response_length)
                self.vocab[env_id] = vocab
            current_group += groups

    def render(self, outputs, *, training, history_windows=None):
        texts, conversations = [], []
        for output in outputs:
            env_id = output['env_id']
            history = output['history']
            if training and history and 'state' in history[-1]:
                history = history[:-1]
            first_turn = 0
            if history_windows is not None:
                if training:
                    raise ValueError('training retains the complete genuine history')
                window = history_windows[env_id]
                if type(window) is not int or not 0 <= window < len(history):
                    raise ValueError('invalid number of prior turns')
                first_turn = len(history) - 1 - window
                history = history[first_turn:]
            messages = [{'role': 'system', 'content': "You're a helpful assistant. "},
                        {'role': 'user', 'content': self.prefixes[env_id]}]
            for local_idx, turn in enumerate(history):
                idx = local_idx + first_turn
                messages[-1]['content'] += f'\nTurn {idx + 1}:\n'
                if 'state' in turn:
                    fmt = '<think> [Your thoughts] </think> <answer> [your answer] </answer>' if self.enable_think else '<answer> [your answer] </answer>'
                    length = f'Max response length: {self.max_tokens[env_id]} words (tokens).'
                    messages[-1]['content'] += f'State:\n{turn["state"]}\nYou have {turn["actions_left"]} actions left. Always output: {fmt} with no extra text. Strictly follow this format. {length}\n'
                if 'llm_response' in turn:
                    messages.append({'role': 'assistant', 'content': turn['llm_response']})
                if 'reward' in turn and not (training and local_idx == len(history) - 1):
                    messages.append({'role': 'user', 'content': f'Reward:\n{turn["reward"]}\n'})
            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=not training, tokenize=False)
            if not training:
                text += '<think>' if self.enable_think else '<answer>'
            texts.append(text)
            conversations.append(messages)
        return texts, conversations

    def validate_tokens(self, outputs, batch, *, training):
        reason = 'assistant_provenance' if training else 'prompt_provenance'
        try:
            texts, _ = self.render(outputs, training=training)
            expected = self.tokenizer(texts, return_tensors='pt', padding=True, padding_side='left', truncation=False)
            ids, masks = _list(batch.batch['input_ids']), _list(batch.batch['attention_mask'])
            target_ids, target_masks = _list(expected.input_ids), _list(expected.attention_mask)
            env_ids = _list(batch.non_tensor_batch['env_ids'])
            groups = _list(batch.non_tensor_batch['group_ids'])
            raw_prompts = batch.non_tensor_batch.get('raw_prompt_ids')
            if not training and ('multi_modal_data' in batch.non_tensor_batch
                    or (raw_prompts is not None and len(raw_prompts) != len(outputs))):
                raise ProvenanceError(reason)
            if env_ids != [r['env_id'] for r in outputs] or groups != [r['group_id'] for r in outputs]:
                raise ProvenanceError(reason)
            if len(ids) != len(outputs) or len(masks) != len(outputs):
                raise ProvenanceError(reason)
            for index, (output, row, mask, target, target_mask) in enumerate(zip(outputs, ids, masks, target_ids, target_masks)):
                if len(row) != len(mask) or any(m not in (0, 1) for m in mask):
                    raise ProvenanceError(reason)
                if any(a > b for a, b in zip(mask, mask[1:])):
                    raise ProvenanceError(reason)
                if any(t != self.tokenizer.pad_token_id for t, m in zip(row, mask) if not m):
                    raise ProvenanceError(reason)
                visible = [t for t, m in zip(row, mask) if m]
                allowed = [[t for t, m in zip(target, target_mask) if m]]
                if not training:
                    # Context selection is allowed, invented content is not. Each
                    # option is rendered from a suffix of the trusted history.
                    for window in range(len(output['history']) - 1):
                        text, _ = self.render([output], training=False, history_windows={output['env_id']: window})
                        tokenized = self.tokenizer(text, return_tensors='pt', padding=True, padding_side='left', truncation=False)
                        allowed.append([t for t, m in zip(_list(tokenized.input_ids)[0], _list(tokenized.attention_mask)[0]) if m])
                if visible not in allowed:
                    raise ProvenanceError(reason)
                # The actual vLLM path gives this optional field precedence.
                if not training and raw_prompts is not None and _list(raw_prompts[index]) != visible:
                    raise ProvenanceError(reason)
        except (KeyError, TypeError, AttributeError, IndexError) as error:
            raise ProvenanceError(reason) from error

    def actual_inputs(self, outputs):
        if outputs.batch is not None and 'responses' in outputs.batch:
            generated = self.tokenizer.batch_decode(outputs.batch['responses'], skip_special_tokens=True)
        else:
            generated = outputs.non_tensor_batch['response_texts']
        env_ids = _list(outputs.non_tensor_batch['env_ids'])
        if len(generated) != len(env_ids):
            raise ProvenanceError('action_provenance')
        result = []
        for env_id, text in zip(env_ids, generated):
            # The fixed generation prefix is part of the actual assistant turn.
            raw = ('<think>' if self.enable_think else '<answer>') + text
            canonical, actions = parse_response(raw, enable_think=self.enable_think,
                action_sep=self.action_sep, max_actions=self.max_actions)
            result.append({'env_id': env_id, 'llm_raw_response': raw,
                           'llm_response': canonical, 'actions': actions})
        return result

    def validate_actions(self, actual, candidate):
        if type(candidate) is not list or len(candidate) != len(actual):
            raise ProvenanceError('action_provenance')
        for expected, row in zip(actual, candidate):
            if type(row) is not dict or any(row.get(k) != v for k, v in expected.items()):
                raise ProvenanceError('action_provenance')

    def validate_transition(self, before, actual, manager):
        """Executed actions are the vocabulary-mapped, budget/done-limited prefix."""
        for observation, expected in zip(before, actual):
            env_id = expected['env_id']
            if observation['env_id'] != env_id:
                raise ProvenanceError('action_provenance')
            history = manager.rollout_cache[env_id]['history']
            turn = history[len(observation['history']) - 1]
            mapped = [self.vocab[env_id][a.lower()] for a in expected['actions'] if a.lower() in self.vocab[env_id]]
            allowed = mapped[:observation['history'][-1]['actions_left']]
            executed = turn['actions']
            if executed != allowed[:len(executed)] or len(executed) > len(allowed):
                raise ProvenanceError('action_provenance')
            if len(executed) != len(allowed) and not manager.envs[env_id]['status'].terminated:
                raise ProvenanceError('action_provenance')
            if turn['llm_raw_response'] != expected['llm_raw_response'] or turn['llm_response'] != expected['llm_response']:
                raise ProvenanceError('action_provenance')
