"""Inference"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from itertools import islice
import json
import os
from pathlib import Path
from string import Template
from typing import Callable

from parse_regex_scalpel import Output as ScalpelData
from peft import PeftModel
from scalpel_to_thinking_steps import subpattern_to_string
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig


@dataclass
class Output:
    pattern: str | None
    query: str
    thinking: str
    answer: str


def load_quantized_model(  # type: ignore
    model_name: str,
):
    quantization_config = BitsAndBytesConfig(  # type: ignore
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
    )
    if "gpt-oss" in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            quantization_config=quantization_config,
            device_map="auto",
        )
    return model


ChatType = list[dict[str, str]]


def get_line_to_chat_and_pattern(
    task: str,
) -> Callable[[str, ChatType | None], tuple[ChatType, str] | None]:

    with open("./prompts/prompt.txt", encoding="utf-8") as f:
        simple_template = Template(f.read())
    with open("./prompts/localization.txt", encoding="utf-8") as f:
        localization_template = Template(f.read())
    with open("./prompts/1-shot-prompt.txt", encoding="utf-8") as f:
        one_shot_template = Template(f.read())
    with open("./prompts/5-shot-prompt.txt", encoding="utf-8") as f:
        five_shot_template = Template(f.read())
    with open(
        "./prompts/5-shot-localization-prompt.txt", encoding="utf-8"
    ) as f:
        five_shot_localization_template = Template(f.read())

    def simple_line_to_chat(
        line: str,
        previous_chat: list[dict[str, str]] | None,
    ) -> tuple[ChatType, str] | None:
        if previous_chat is not None:
            return None
        pattern = line.rstrip("\r\n")
        query = simple_template.substitute(pattern=pattern)
        chat = [{"role": "user", "content": query}]
        return chat, pattern

    def one_shot_line_to_chat(
        line: str,
        previous_chat: list[dict[str, str]] | None,
    ) -> tuple[ChatType, str] | None:
        if previous_chat is not None:
            return None
        pattern = line.rstrip("\r\n")
        query = one_shot_template.substitute(pattern=pattern)
        chat = [{"role": "user", "content": query}]
        return chat, pattern

    def five_shot_line_to_chat(
        line: str,
        previous_chat: list[dict[str, str]] | None,
    ) -> tuple[ChatType, str] | None:
        if previous_chat is not None:
            return None
        pattern = line.rstrip("\r\n")
        query = five_shot_template.substitute(pattern=pattern)
        chat = [{"role": "user", "content": query}]
        return chat, pattern

    def localization_line_to_chat(
        line: str,
        previous_chat: list[dict[str, str]] | None,
    ) -> tuple[ChatType, str] | None:
        if previous_chat is not None:
            return None
        entry = ScalpelData.loads(line)
        if entry.pattern is None:
            return simple_line_to_chat("", previous_chat)
        elif entry.result is None:
            return simple_line_to_chat(entry.pattern, previous_chat)
        else:
            subpatterns: list[str] = []
            if entry.result.repairs:
                repair = entry.result.repairs[0]
                if repair.subpatterns is not None:
                    for subpattern in repair.subpatterns:
                        subpatterns.append(subpattern)

            if not subpatterns:
                query = simple_template.substitute(pattern=entry.pattern)
            else:
                query = localization_template.substitute(
                    pattern=entry.pattern,
                    subpatterns=", ".join(
                        map(subpattern_to_string, subpatterns)
                    ),
                )
        chat = [{"role": "user", "content": query}]
        return chat, entry.pattern

    def five_shot_localization_line_to_chat(
        line: str,
        previous_chat: list[dict[str, str]] | None,
    ) -> tuple[ChatType, str] | None:
        if previous_chat is not None:
            return None
        entry = ScalpelData.loads(line)
        if entry.pattern is None:
            return localization_line_to_chat("", previous_chat)
        elif entry.result is None:
            return localization_line_to_chat(entry.pattern, previous_chat)
        else:
            subpatterns: list[str] = []
            if entry.result.repairs:
                repair = entry.result.repairs[0]
                if repair.subpatterns is not None:
                    for subpattern in repair.subpatterns:
                        subpatterns.append(subpattern)

            if not subpatterns:
                query = five_shot_template.substitute(pattern=entry.pattern)
            else:
                query = five_shot_localization_template.substitute(
                    pattern=entry.pattern,
                    subpatterns=", ".join(
                        f"`{subpattern_to_string(e)}`" for e in subpatterns
                    ),
                )
        chat = [{"role": "user", "content": query}]
        return chat, entry.pattern

    if task == "simple":
        return simple_line_to_chat
    elif task == "localization":
        return localization_line_to_chat
    elif task == "one_shot":
        return one_shot_line_to_chat
    elif task == "five_shot":
        return five_shot_line_to_chat
    elif task == "five_shot_localization":
        return five_shot_localization_line_to_chat
    else:
        assert False


def main() -> None:
    parser = argparse.ArgumentParser()

    # Argument
    parser.add_argument("--model-name", default="openai/gpt-oss-20b", type=str)
    parser.add_argument("--reasoning", action="store_true", default=False)
    parser.add_argument(
        "--dataset-path", default="./data/patterns/sl_regexes.txt", type=Path
    )
    parser.add_argument(
        "--output-dir",
        default="./data/large-language-models",
        type=Path,
    )
    parser.add_argument("--task", default="simple", type=str)
    parser.add_argument("--max-think-tokens", default=4096, type=int)
    parser.add_argument("--max-answer-tokens", default=1024, type=int)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", default=None, type=float)
    parser.add_argument("--top-p", default=None, type=float)
    parser.add_argument(
        "--gpt-oss-mode", choices=["high", "medium", "low"], default="low"
    )
    parser.add_argument("--start-idx", default=0, type=int)
    parser.add_argument("--end-idx", default=2147483647, type=int)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--adapter-path", type=Path, default=None)

    args = parser.parse_args()

    if args.start_idx > args.end_idx:
        print("Start index is larger than end index!")
        exit(1)

    # Output Path
    output_dir = args.output_dir / args.task
    result_basename = args.model_name.split("/")[-1].replace("-", "_")
    if "gpt-oss" in args.model_name.lower():
        result_basename += f"-{args.gpt_oss_mode}"
    if args.reasoning:
        result_basename += "-reasoning"
    else:
        result_basename += "-noreasoning"
    if args.do_sample:
        result_basename += f"-{args.temperature}"
        result_basename += f"-{args.top_p}"
    if args.start_idx > 0:
        result_basename += f"-{args.start_idx}"
    if args.tag:
        result_basename += f"-{args.tag}"
    result_basename += ".jsonl"
    result_path = output_dir / result_basename
    os.makedirs(output_dir, exist_ok=True)

    # Loading Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # type: ignore

    think_end_token: str | None = None
    eos_token_id = None

    if "exaone" in args.model_name.lower():
        apply_chat_template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.reasoning,
        )
    elif "qwen" in args.model_name.lower():
        apply_chat_template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.reasoning,
        )
    elif "gpt-oss" in args.model_name.lower():
        apply_chat_template_kwargs = dict(
            reasoning_effort=args.gpt_oss_mode,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        apply_chat_template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
        )

    model = load_quantized_model(args.model_name)
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    assert model.generation_config is not None

    # Setting Config and Variables for Reasoning
    thinking_generation_config = None
    if args.reasoning:
        if "gpt-oss" in args.model_name:
            think_end_token = "<|end|>"
        else:
            think_end_token = "</think>"
        encoded_think_end_token = tokenizer.encode(
            think_end_token, add_special_tokens=False
        )

        thinking_generation_config = deepcopy(model.generation_config)
        thinking_generation_config.max_new_tokens = args.max_think_tokens
        if len(encoded_think_end_token) == 1:
            eos_token_id = encoded_think_end_token[0]
            thinking_generation_config.eos_token_id = eos_token_id
        else:
            thinking_generation_config.stop_strings = think_end_token

    # Setting Config for Sampling
    model.generation_config.do_sample = args.do_sample
    if args.do_sample:
        if args.temperature is not None:
            model.generation_config.temperature = args.temperature
        if args.top_p is not None:
            model.generation_config.top_p = args.top_p

    # Setting Maximum Answer Tokens
    answer_generation_config = deepcopy(model.generation_config)
    answer_generation_config.max_new_tokens = args.max_answer_tokens

    # Setting line_to_chat
    line_to_chat_and_pattern = get_line_to_chat_and_pattern(args.task)

    # Start Inferencing
    with (
        open(args.dataset_path, "r", encoding="utf-8") as input_file,
        open(result_path, "a", encoding="utf-8") as output_file,
    ):

        for _, line in islice(
            enumerate(tqdm(input_file)), args.start_idx, args.end_idx
        ):
            chat_and_pattern = line_to_chat_and_pattern(line, None)

            assert chat_and_pattern is not None
            chat, pattern = chat_and_pattern
            input_template = tokenizer.apply_chat_template(
                chat,
                **apply_chat_template_kwargs,
            )

            inputs = tokenizer(
                input_template,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
            query_length = len(inputs["input_ids"][0])

            # Reasoning Inference
            thinking_length = 0
            if thinking_generation_config is not None:
                output_ids = model.generate(
                    **inputs,
                    generation_config=thinking_generation_config,
                    tokenizer=tokenizer if eos_token_id is None else None,
                )[0]
                output_string = tokenizer.decode(output_ids[query_length:])
                inputs = tokenizer(
                    input_template
                    + output_string.split(think_end_token)[0]
                    + think_end_token,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).to("cuda")
                thinking_length = len(inputs["input_ids"][0]) - query_length

            # Answer Inference
            output_ids = model.generate(
                **inputs,
                generation_config=answer_generation_config,
                tokenizer=tokenizer,
            )[0]

            query = tokenizer.decode(output_ids[:query_length])
            thinking = tokenizer.decode(
                output_ids[query_length : query_length + thinking_length]
            )
            answer = tokenizer.decode(
                output_ids[query_length + thinking_length :]
            )

            result = Output(
                pattern=pattern,
                query=query,
                thinking=thinking,
                answer=answer,
            )

            # Saving Output
            json.dump(asdict(result), output_file, ensure_ascii=False)
            output_file.write("\n")


if __name__ == "__main__":
    main()
