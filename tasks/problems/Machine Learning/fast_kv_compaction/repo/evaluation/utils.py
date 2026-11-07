# evaluation/utils.py
"""
Shared utilities for evaluation tasks.

This module contains common functionality used across different evaluation scripts:
- Model loading
- KV cache extraction
- Chat template formatting
- vLLM initialization and management
- Answer parsing utilities
"""
import re
import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from models.generate import get_generation_params

QWEN_USER_START = "<|im_start|>user"
QWEN_USER_END = "<|im_end|>"
LLAMA_USER_START = "<|start_header_id|>user"
LLAMA_USER_END = "<|eot_id|>"
GEMMA_USER_START = "<start_of_turn>user"
GEMMA_USER_END = "<end_of_turn>"


def detect_user_tags(formatted_context: str) -> Tuple[str, str]:
    """
    Detect the user start/end tags used in the formatted chat context.

    Parameters
    ----------
    formatted_context : str
        Chat-formatted string produced by tokenizer.apply_chat_template.

    Returns
    -------
    tuple
        (user_start_tag, user_end_tag)
    """
    if LLAMA_USER_START in formatted_context:
        return LLAMA_USER_START, LLAMA_USER_END
    if QWEN_USER_START in formatted_context:
        return QWEN_USER_START, QWEN_USER_END
    if GEMMA_USER_START in formatted_context:
        return GEMMA_USER_START, GEMMA_USER_END
    raise ValueError("Could not detect user tags from formatted context.")


def compute_article_indices(
    tokenizer,
    formatted_context: str,
    article_text: str,
) -> range:
    """
    Compute the token indices of the article portion within a formatted context.

    This function finds where the article text appears within the chat-formatted
    context and returns the corresponding token index range.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode the context
    formatted_context : str
        Chat-formatted string produced by tokenizer.apply_chat_template
    article_text : str
        The original article text (before chat formatting)

    Returns
    -------
    range
        Token indices corresponding to the article portion (start_idx, end_idx)
    """
    # Detect user tags to find article boundaries
    user_start_tag, _ = detect_user_tags(formatted_context)

    # Find position after the user start tag
    user_start_pos = formatted_context.find(user_start_tag)
    if user_start_pos == -1:
        raise ValueError(f"Could not find '{user_start_tag}' tag in formatted context")

    # The article starts after the user tag and a newline
    article_text_start = formatted_context.find('\n', user_start_pos + len(user_start_tag))
    if article_text_start == -1:
        raise ValueError("Could not find newline after user start tag")
    article_text_start += 1  # Skip the newline itself

    # The article ends at article_text_start + len(article_text)
    article_text_end = article_text_start + len(article_text)

    # Tokenize prefix (everything before article) to find article start token
    prefix_text = formatted_context[:article_text_start]
    prefix_tokens = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
    article_start_idx = prefix_tokens['input_ids'].shape[1]

    # Tokenize everything up to and including the article to find article end token
    article_with_prefix = formatted_context[:article_text_end]
    article_with_prefix_tokens = tokenizer(article_with_prefix, return_tensors="pt", add_special_tokens=False)
    article_end_idx = article_with_prefix_tokens['input_ids'].shape[1]

    return range(article_start_idx, article_end_idx)


def load_model_and_tokenizer(
    model_name: str,
    device: str = None,
    dtype: Optional[torch.dtype] = None,
    max_model_len: Optional[int] = None,
):
    """
    Load model and tokenizer with proper configuration.

    Parameters
    ----------
    model_name : str
        HuggingFace model name
    device : str, optional
        Device to use ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for computations
    max_model_len : int, optional
        Maximum model context length. If specified, enables YaRN rope scaling
        to handle extended context lengths beyond the model's default.

    Returns
    -------
    model : PreTrainedModel
        Loaded model
    tokenizer : PreTrainedTokenizer
        Loaded tokenizer
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = {
        "device_map": device if device == "cuda" else None,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa"
    }
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    # Enable YaRN rope scaling for extended context if max_model_len is specified
    if max_model_len is not None:
        # Determine the model's native max position embeddings
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        original_max_pos = getattr(config, 'max_position_embeddings', 32768)
        if max_model_len > original_max_pos:
            model_kwargs["rope_scaling"] = {
                "rope_type": "yarn",
                "factor": max_model_len / original_max_pos,
                "original_max_position_embeddings": original_max_pos,
            }
            print(f"Enabling YaRN rope scaling: {original_max_pos} -> {max_model_len} (factor={max_model_len/original_max_pos:.1f}x)")

    # Use Qwen3ForCausalLM for qwen3 models, otherwise use AutoModelForCausalLM
    if "qwen" in model_name.lower():
        from models.qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif "llama" in model_name.lower():
        from models.llama import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    elif "gemma" in model_name.lower():
        from models.gemma3 import Gemma3ForCausalLM
        model = Gemma3ForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
    else:
        print("Warning: Using AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    if device == "cpu":
        model = model.to(device)

    model.eval()
    print(f"Model loaded successfully")
    return model, tokenizer


def initialize_vllm(
    model_name: str,
    gpu_memory_utilization: float = 0.5,
    max_model_len: Optional[int] = None,
):
    """
    Initialize vLLM model for query generation.

    Parameters
    ----------
    model_name : str
        HuggingFace model name to use with vLLM
    gpu_memory_utilization : float
        Fraction of GPU memory to use (default: 0.5)
    max_model_len : int, optional
        Maximum model context length. If None, uses model default.
        For long contexts, set to 131072 or higher.
        When set, also enables YaRN rope scaling for extended context.

    Returns
    -------
    vllm_model : LLM
        Initialized vLLM model in sleep mode
    """
    print("Initializing vLLM model (will be reused across all articles)...")

    # Set vLLM multiprocessing method to 'spawn' to avoid CUDA fork issues
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vLLM is required for this run."
        )

    # Build kwargs for LLM initialization
    llm_kwargs = {
        "model": model_name,
        "tokenizer": model_name,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_sleep_mode": True,
    }

    # Set max_model_len and rope_scaling if specified
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
        # Enable YaRN rope scaling if extending beyond native context
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        original_max_pos = getattr(config, 'max_position_embeddings', 32768)
        if max_model_len > original_max_pos:
            llm_kwargs["hf_overrides"] = {
                "rope_scaling": {
                    "rope_type": "yarn",
                    "factor": max_model_len / original_max_pos,
                    "original_max_position_embeddings": original_max_pos,
                }
            }
            print(f"Using extended context: max_model_len={max_model_len} with YaRN rope scaling (factor={max_model_len/original_max_pos:.1f}x)")
        else:
            print(f"Setting max_model_len={max_model_len} (within native {original_max_pos})")

    vllm_model = LLM(**llm_kwargs)
    # Put it to sleep immediately to free GPU memory
    vllm_model.sleep()
    print(f"vLLM initialized and put to sleep (model: {model_name})")

    return vllm_model


def extract_full_kv_cache(
    model,
    tokenizer,
    article_text: str,
    device: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Tuple[int, Tuple, range, str, int]:
    """
    Extract full KV cache from article text formatted with QA chat template.

    Parameters
    ----------
    model : PreTrainedModel
        The model to use for cache extraction
    tokenizer : PreTrainedTokenizer
        The tokenizer
    article_text : str
        The article text
    device : str
        Device to use
    system_prompt : str, optional
        System prompt to use for formatting. If None, uses default based on model_name.
    model_name : str, optional
        Model name to determine default system prompt (gemma uses empty prompt).

    Returns
    -------
    seq_len : int
        Sequence length of the formatted context
    past_key_values : tuple
        Full KV cache from the model
    article_indices : range
        Indices of the article portion (between <user> and </user> tags)
    formatted_context : str
        The formatted context string (article with chat template applied)
    original_token_length : int
        Original token length of the sequence
    """
    # Format article as context using QA-specific system prompt
    formatted_context = format_context(tokenizer, article_text, system_prompt, model_name)

    # Tokenize the full context
    # Use add_special_tokens=False since formatted_context already has <bos> from chat template
    inputs = tokenizer(formatted_context, return_tensors="pt", add_special_tokens=False)
    original_token_length = inputs['input_ids'].shape[1]

    input_ids = inputs['input_ids'].to(device)
    seq_len = input_ids.shape[1]

    # Compute article token indices using the shared utility function
    article_indices = compute_article_indices(tokenizer, formatted_context, article_text)

    with torch.no_grad():
        outputs = model(
            input_ids,
            use_cache=True
        )
        # TODO: fix the below for models with sliding layers 
        # from models.generate import chunked_prefill
        # outputs = chunked_prefill(
        #     model,
        #     input_ids=input_ids,
        # )
    return seq_len, outputs.past_key_values, article_indices, formatted_context, original_token_length


def get_default_system_prompt(model_name: Optional[str] = None) -> str:
    """
    Get the default system prompt for a model.

    Gemma models don't have explicit system prompts.
    """
    if model_name and "gemma" in model_name.lower():
        return ""
    return "You are a helpful assistant. Answer a question based on the provided context."


def format_context(
    tokenizer,
    article: str,
    system_prompt: Optional[str] = None,
    model_name: Optional[str] = None,
) -> str:
    """
    Format article as context using QA-specific system prompt.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer
    article : str
        The article text
    system_prompt : str, optional
        System prompt to use. If None, uses default based on model_name.
    model_name : str, optional
        Model name to determine default system prompt (gemma uses none).

    Returns
    -------
    str
        Formatted context
    """
    if system_prompt is None:
        system_prompt = get_default_system_prompt(model_name)

    # For gemma models, omit the system message entirely
    if system_prompt == "":
        context_messages = [
            {"role": "user", "content": article}
        ]
    else:
        context_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": article}
        ]
    return tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )


def format_question(
    tokenizer,
    question: str,
    options: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    enable_thinking: bool = True,
    answer_prefix: Optional[str] = None,
) -> str:
    """
    Format question as prompt using chat template.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer
    question : str
        The question text
    options : list of str, optional
        Multiple choice options. If provided, formats with letters (A, B, C, D, E, etc.)
    model_name : str, optional
        Model name to determine prompt format (used to detect Llama models)
    enable_thinking : bool
        Whether to enable thinking mode in the chat template (default: True).
        Set to False for tasks like RULER where thinking tokens would interfere
        with string-match scoring.
    answer_prefix : str, optional
        If provided, appended after the generation prompt to guide the model's response.
        Used by RULER tasks (e.g., "The special magic number for X is").

    Returns
    -------
    str
        Formatted prompt ready for generation
    """
    # Detect model type for special handling
    is_qwen = model_name and "qwen" in model_name.lower()
    is_gemma = model_name and "gemma" in model_name.lower()

    if options:
        # Format as multiple choice with A, B, C, D, E, etc.
        # Support up to 26 options (A-Z)
        num_options = len(options)
        option_labels = [chr(65 + i) for i in range(num_options)]  # 65 is 'A'
        formatted_options = '\n'.join([
            f"{label}. {option}"
            for label, option in zip(option_labels, options)
        ])

        # Generate instruction based on number of options
        if num_options <= 4:
            option_list = "A, B, C, or D"
        elif num_options == 5:
            option_list = "A, B, C, D, or E"
        else:
            option_list = ", ".join(option_labels[:-1]) + f", or {option_labels[-1]}"

        if is_qwen:
            if model_name == "Qwen/Qwen3-4B-Instruct-2507": # need to specify step by step
                instruction = f"Please think step by step very briefly and then, on a new line, respond with the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
            else:
                instruction = f"Please think very briefly and then respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        elif is_gemma: # need to remove "briefly" so it actually thinks...
            instruction = f"Please think step by step and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."
        else:
            instruction = f"Please think step by step briefly and then, on a new line, respond with **only** the letter ({option_list}) of the correct option. If you are not sure, still make a guess."

        content = f"{question}\n\n{formatted_options}\n\n{instruction}"
    else:
        content = question

    question_messages = [
        {"role": "user", "content": content}
    ]

    result = tokenizer.apply_chat_template(
        question_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    # Strip the first <bos> token for Gemma models
    if is_gemma:
        result = result[len("<bos>"):]

    # Append answer prefix for guided generation (e.g., RULER tasks)
    if answer_prefix:
        result = result + answer_prefix

    return result


def parse_model_choice(answer_text: str, max_options: int = 5) -> Optional[int]:
    """
    Parse the model's answer to extract the chosen option index.

    Handles thinking mode where answer may be in format:
    <think>reasoning...</think>
    A

    Also handles formats like:
    - "the answer is: D."
    - "the answer is D"

    Parameters
    ----------
    answer_text : str
        The generated answer text
    max_options : int
        Maximum number of options (default: 5 for A-E)

    Returns
    -------
    int or None
        The option index (1=A, 2=B, 3=C, 4=D, 5=E, ...) or None if cannot parse
    """
    has_open_think = '<think>' in answer_text
    has_close_think = '</think>' in answer_text

    # Incomplete thinking tag => can't parse yet
    if has_open_think and not has_close_think:
        return None

    if has_close_think:
        parts = answer_text.split('</think>')
        if len(parts) > 1:
            answer_text = parts[-1].strip()
    elif not has_open_think:
        # No <think> at all means take the last non-empty line
        lines = answer_text.splitlines()
        for line in reversed(lines):
            if line.strip():
                answer_text = line.strip()
                break

    # Try to extract from various answer formats:
    # - "the answer is: X" or "the answer is X"
    # - "Answer: X" or "Answer X"
    answer_lower = answer_text.lower()
    for pattern in ['answer:', 'answer is', 'correct option is']:
        if pattern in answer_lower:
            # Extract everything after the pattern
            parts = answer_lower.split(pattern)
            if len(parts) > 1:
                # Remove colons, periods, *, and whitespace
                candidate = parts[-1].strip().lstrip(':').strip().rstrip('.').strip()
                candidate = candidate.replace('*', '')
                if candidate and candidate[0] in tuple(chr(97 + i) for i in range(max_options)):  # a-e (or more)
                    return ord(candidate[0]) - ord('a') + 1

    answer_clean = answer_lower.split(".")[0].strip()
    answer_clean = answer_clean.replace('*', '')

    # Check that it matches valid options (a-e by default, or more)
    # Return 1-indexed to match dataset gold_label format
    valid_options = tuple(chr(97 + i) for i in range(max_options))  # a, b, c, d, e, ...
    if answer_clean in valid_options:
        return ord(answer_clean) - ord('a') + 1

    # Fallback: look for standalone capital letters (A, B, C, D, E) in the last line
    valid_upper = tuple(chr(65 + i) for i in range(max_options))  # A, B, C, D, E, ...
    # Search for standalone letter (word boundary or surrounded by non-letters)
    pattern = r'(?<![a-zA-Z])([' + ''.join(valid_upper) + r'])(?![a-zA-Z])'
    matches = re.findall(pattern, answer_text)
    if matches:
        # Take the last match (most likely to be the final answer)
        letter = matches[-1]
        return ord(letter) - ord('A') + 1

    return None


