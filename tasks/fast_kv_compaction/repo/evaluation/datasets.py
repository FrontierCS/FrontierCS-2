# evaluation/datasets.py
"""
Dataset loaders for evaluation tasks.

This module provides loaders for various QA and evaluation datasets.
Each loader returns a standardized format for use with evaluators.

Standardized Format
-------------------
All loaders return a list of articles/documents with the following structure:
    - article_id: Unique identifier
    - title: Article/document title or name
    - article: Full text content (concatenated if multiple notes)
    - questions: List of questions, each with:
        - question: Question text
        - options: List of answer options (4 for QuALITY, 5 for LongHealth)
        - gold_label: Correct answer index (1-indexed: 1=A, 2=B, 3=C, 4=D, 5=E)
        - question_unique_id: Unique question identifier
"""
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download


def load_quality_data(data_path: str) -> List[Dict]:
    """
    Load QuALITY dataset from JSONL file.

    The QuALITY dataset contains long articles with multiple-choice questions (4 options).

    Parameters
    ----------
    data_path : str
        Path to QuALITY dataset JSON file (JSONL format)

    Returns
    -------
    data : list of dict
        List of articles in standardized format
    """
    print(f"Loading QuALITY data from: {data_path}")
    combined_articles = {}
    total_entries = 0

    with open(data_path, 'r') as f:
        for line in f:
            if line.strip():
                total_entries += 1
                article = json.loads(line)
                original_article_id = article['article_id']
                article['article'] = article.get('article', '').strip()
                key = (original_article_id, article['article'])

                # Ensure questions list exists even if missing
                questions = article.get('questions', [])

                # Prepend dataset name to article_id for downstream uniqueness
                article['article_id'] = f"quality_{original_article_id}"
                article['questions'] = questions

                if key in combined_articles:
                    combined_articles[key]['questions'].extend(questions)
                else:
                    combined_articles[key] = article

    data = list(combined_articles.values())
    print(f"Loaded {len(data)} unique articles (from {total_entries} entries)")
    return data


def load_longhealth_data(
    data_path: str,
    include_diagnosis: bool = True,
    patients_per_article: int = 1
) -> List[Dict]:
    """
    Load LongHealth dataset from JSON file.

    The LongHealth dataset contains patient medical records with multiple-choice questions (5 options).
    Always loads ALL patients in sorted order.

    Parameters
    ----------
    data_path : str
        Path to LongHealth dataset JSON file
    include_diagnosis : bool
        Whether to include diagnosis in the article text (default: True)
    patients_per_article : int
        Number of patients to group into each article (default: 1).
        Creates floor(n/patients_per_article) articles.
        When > 1, questions are prefixed with patient info.

    Returns
    -------
    data : list of dict
        List of patient records in standardized format, each containing:
        - article_id: Patient ID (or group ID if patients_per_article > 1)
        - title: Patient name (or group title if patients_per_article > 1)
        - article: Concatenated medical notes
        - questions: List of questions with 5 options (A-E)
    """
    print(f"Loading LongHealth data from: {data_path}")

    with open(data_path, 'r') as f:
        raw_data = json.load(f)

    # Sort patient IDs to ensure consistent ordering
    sorted_patient_ids = sorted(raw_data.keys())

    num_articles = len(sorted_patient_ids) // patients_per_article
    data = []

    for article_idx in range(num_articles):
        # Get patient IDs for this article
        start_idx = article_idx * patients_per_article
        end_idx = start_idx + patients_per_article
        group_patient_ids = sorted_patient_ids[start_idx:end_idx]

        # Concatenate all patients' medical notes
        article_parts = []
        all_questions = []
        patient_infos = []

        for patient_id in group_patient_ids:
            patient_data = raw_data[patient_id]

            # Add patient medical notes
            texts = patient_data['texts']
            for note_id, note_text in texts.items():
                article_parts.append(f"<{note_id}>\n{note_text}\n</{note_id}>")

            # Store patient info for question prefixing
            if include_diagnosis:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}, Diagnosis: {patient_data['diagnosis']}"
            else:
                patient_info = f"ID {patient_id}, Name: {patient_data['name']}, Birthday: {patient_data['birthday']}"

            patient_infos.append((patient_id, patient_info, patient_data))

        # Combine all article parts
        article = "\n\n".join(article_parts).strip()

        # Process questions from all patients
        for patient_id, patient_info, patient_data in patient_infos:
            for q_idx, q in enumerate(patient_data['questions']):
                # Map answer letters to text
                options = [
                    q['answer_a'],
                    q['answer_b'],
                    q['answer_c'],
                    q['answer_d'],
                    q['answer_e']
                ]

                # Find the gold label index
                correct_answer = q['correct']
                gold_label_idx = None

                for i, opt in enumerate(options):
                    if opt.strip() == correct_answer.strip():
                        gold_label_idx = i + 1
                        break

                if gold_label_idx is None:
                    print(f"Warning: Could not match correct answer '{correct_answer}' to options for {patient_id}, Q{q_idx}")
                    gold_label_idx = 1

                # Prefix question with patient info only if multiple patients per article
                if patients_per_article > 1:
                    prefixed_question = (
                        f"Please answer the question below about the following patient: "
                        f"{patient_info}\n\n{q['question']}"
                    )
                else:
                    prefixed_question = q['question']

                all_questions.append({
                    'question': prefixed_question,
                    'options': options,
                    'gold_label': gold_label_idx,
                    'question_unique_id': f"{patient_id}_q{q.get('No', q_idx)}",
                })

        # Create grouped article entry
        if patients_per_article == 1:
            # Single patient: use original format
            patient_id = group_patient_ids[0]
            patient_data = raw_data[patient_id]
            article_entry = {
                'article_id': f"longhealth_{patient_id}",
                'title': patient_data['name'],
                'article': article,
                'questions': all_questions,
            }
        else:
            # Multiple patients: use grouped format
            article_id = f"longhealth_group_{article_idx:02d}_patients_{start_idx+1:02d}-{end_idx:02d}"
            title = f"Patients {start_idx+1}-{end_idx}"
            article_entry = {
                'article_id': article_id,
                'title': title,
                'article': article,
                'questions': all_questions,
            }

        data.append(article_entry)

    print(f"Loaded {len(data)} grouped articles ({patients_per_article} patients per article)")
    return data


def load_longcodeqa_data(context_length: str = '32K') -> List[Dict]:
    """
    Load LongCodeQA dataset from HuggingFace.

    The LongCodeQA dataset contains code repositories with multiple-choice questions (4 options).
    Data is downloaded from HuggingFace and cached locally.

    Parameters
    ----------
    context_length : str
        Context length variant to load. Options: '32K', '64K', '128K', '256K', '512K', '1M'

    Returns
    -------
    data : list of dict
        List of articles in standardized format
    """
    valid_lengths = ['32K', '64K', '128K', '256K', '512K', '1M']
    if context_length not in valid_lengths:
        raise ValueError(f"Invalid context_length '{context_length}'. Must be one of: {valid_lengths}")

    print(f"Loading LongCodeQA {context_length} from HuggingFace...")

    # Download the zip file (cached by huggingface_hub)
    zip_path = hf_hub_download(
        repo_id='Steefano/LCB',
        filename='LongCodeQA.zip',
        repo_type='dataset'
    )

    # Read the specific JSON file from the zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(f'LQA/{context_length}.json') as f:
            raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} entries from {context_length}.json")

    # Group entries by repo_text (same article may have multiple questions)
    combined_articles: Dict[str, Dict] = {}
    letter_to_idx = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    for idx, entry in enumerate(raw_data):
        repo_text = entry['repo_text']
        repo_name = entry['repo'].replace('/', '_')

        # Parse options from the question text
        question_text = entry['question']
        options = _parse_options_from_question(question_text)
        question_only = _extract_question_text(question_text)
        gold_label = letter_to_idx.get(entry['correct_letter'], 1)

        question_entry = {
            'question': question_only,
            'options': options,
            'gold_label': gold_label,
            'question_unique_id': f"lqa_{context_length}_{repo_name}_q{idx}",
        }

        if repo_text in combined_articles:
            # Add question to existing article
            combined_articles[repo_text]['questions'].append(question_entry)
        else:
            # Create new article entry
            combined_articles[repo_text] = {
                'article_id': f"lqa_{context_length}_{repo_name}",
                'title': entry['repo'],
                'article': repo_text.strip(),
                'questions': [question_entry],
            }

    data = list(combined_articles.values())
    total_questions = sum(len(article['questions']) for article in data)
    print(f"Converted to {len(data)} unique articles with {total_questions} total questions")
    return data


def _parse_options_from_question(question_text: str) -> List[str]:
    """Parse A), B), C), D) options from question text."""
    # Match patterns like "A) some text" or "A. some text"
    pattern = r'([A-D])[)\.]\s*(.+?)(?=(?:[A-D][)\.]|\Z))'
    matches = re.findall(pattern, question_text, re.DOTALL)

    if len(matches) == 4:
        # Return just the option text, stripped
        return [match[1].strip() for match in matches]

    # Fallback: try to split by newlines with letter prefixes
    options = []
    for letter in ['A', 'B', 'C', 'D']:
        pattern = rf'{letter}[)\.]\s*(.+?)(?:\n|$)'
        match = re.search(pattern, question_text, re.DOTALL)
        if match:
            options.append(match.group(1).strip())

    if len(options) == 4:
        return options

    # Last resort: return empty options (will need manual inspection)
    print(f"Warning: Could not parse 4 options from question")
    return ['Option A', 'Option B', 'Option C', 'Option D']


def _extract_question_text(question_text: str) -> str:
    """Extract just the question part, before the A) B) C) D) options."""
    # Find where the options start (first occurrence of "A)" or "A.")
    match = re.search(r'\nA[)\.]\s', question_text)
    if match:
        return question_text[:match.start()].strip()
    return question_text.strip()


def load_longsweb_data(context_length: str = '64K') -> List[Dict]:
    """
    Load LongSWE-bench dataset from HuggingFace.

    The LongSWE-bench dataset contains long-context software engineering tasks
    where models must understand a large code context and produce a patch.
    This loader is designed for perplexity evaluation on the ground truth patch,
    NOT for generation.

    Unlike the multiple-choice datasets, each entry has:
    - article: The long code context (from 'text' field)
    - question: The problem statement
    - ground_truth: The gold patch to evaluate perplexity on

    Parameters
    ----------
    context_length : str
        Context length variant to load. Options: '32K', '64K', '128K', '256K', '512K', '1M'.
        Default is '64K'.

    Returns
    -------
    data : list of dict
        List of articles in format suitable for perplexity evaluation:
        - article_id: Unique identifier (instance_id)
        - title: Repository name
        - article: The long code context
        - questions: List with single entry containing:
            - question: The problem statement
            - ground_truth: The gold patch (for perplexity evaluation)
            - question_unique_id: Unique identifier
    """
    import os
    import tempfile
    from datasets import DatasetDict

    print(f"Loading LongSWE-bench ({context_length}) from HuggingFace...")

    context_length_upper = context_length.upper()

    # Download the zip file from HuggingFace
    zip_path = hf_hub_download(repo_id='Steefano/LCB', filename='LongSWE_Bench.zip', repo_type='dataset')

    # Extract and load the dataset for the specified context length
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)

        dataset_path = os.path.join(tmpdir, 'LongSWE_Bench', context_length_upper)
        if not os.path.exists(dataset_path):
            available = [d for d in os.listdir(os.path.join(tmpdir, 'LongSWE_Bench'))
                        if os.path.isdir(os.path.join(tmpdir, 'LongSWE_Bench', d))]
            raise ValueError(f"Context length '{context_length_upper}' not found. Available: {available}")

        dataset_dict = DatasetDict.load_from_disk(dataset_path)
        dataset = dataset_dict['test']

    print(f"Loaded {len(dataset)} entries from Steefano/LCB LongSWE_Bench/{context_length_upper}")

    # Deduplicate by instance_id (dataset may have duplicates)
    seen_ids = set()
    data = []
    for entry in dataset:
        instance_id = entry['instance_id']
        if instance_id in seen_ids:
            continue
        seen_ids.add(instance_id)

        text = entry['text']

        # Extract code from <code>...</code> tags
        code_match = re.search(r'<code>\s*(.*?)\s*</code>', text, re.DOTALL)
        if code_match:
            article_text = code_match.group(1).strip()
        else:
            article_text = text.strip()

        # Extract the patch example and instructions from the original text
        patch_instructions_match = re.search(
            r'(Here is an example of a patch file.*?Respond below:)',
            text, re.DOTALL
        )
        if patch_instructions_match:
            patch_instructions = patch_instructions_match.group(1).strip()
        else:
            raise ValueError(f"Could not find patch instructions in text for instance {instance_id}")

        # Build the question with issue and patch instructions
        problem_statement = entry['problem_statement']
        question_text = f'''Above is a partial code base. You will be provided an issue statement explaining a problem to resolve.
<issue>
{problem_statement}
</issue>

{patch_instructions}'''

        title = f"{entry['repo']} ({instance_id})"

        article_entry = {
            'article_id': f"longsweb_{instance_id}",
            'title': title,
            'article': article_text,
            'questions': [{
                'question': question_text,
                'ground_truth': entry['patch'],
                'question_unique_id': f"longsweb_{instance_id}",
            }],
        }
        data.append(article_entry)

    total_questions = sum(len(article['questions']) for article in data)
    print(f"Converted to {len(data)} unique articles with {total_questions} total questions")
    return data


LONGBENCHV2_CRITICAL_IDS = {
    '671b3cabbb02136c067d5252', '66f958b3bb02136c067c5219', '66f3c1ab821e116aacb2ead1',
    '66fcf2f2bb02136c067c9169', '670fb813bb02136c067d2bec', '6703a0ecbb02136c067cd11b',
    '66ebd5125a08c7b9b35e0616', '66fbab85bb02136c067c81dc', '66ec3aff821e116aacb1c52e',
    '66ec3d1d821e116aacb1c622', '66f55d66821e116aacb33734', '66eae4de5a08c7b9b35dd12d',
    '66ec41d3821e116aacb1c874', '66f3918f821e116aacb2d8b7', '66fba2bcbb02136c067c8112',
    '66f3ad93821e116aacb2e29f', '66ebd0ea5a08c7b9b35dff57', '6719b96abb02136c067d4358',
    '66f0ed5a821e116aacb265ab', '66f37eb9821e116aacb2d295', '66f53f2b821e116aacb3335a',
    '672864b4bb02136c067d916c', '66f7aa19bb02136c067c327e', '66f3fb15821e116aacb303dc',
    '66ed3e90821e116aacb1f82f', '66ebd22e5a08c7b9b35e0126', '671b3e0cbb02136c067d52e5',
    '671b2c05bb02136c067d50c8', '66ebd0825a08c7b9b35dfe9d', '66ec3352821e116aacb1c085',
    '66f59d31821e116aacb340f6', '6704a442bb02136c067cdd91', '66f6bcf3bb02136c067c2703',
    '66f55729821e116aacb3358b', '66f52c6d821e116aacb32cb0', '6725dcb2bb02136c067d85b7',
    '66f68b33bb02136c067c2303', '67286ab8bb02136c067d92e4', '671b08c8bb02136c067d4e19',
    '66f53b9b821e116aacb332fc', '671b080abb02136c067d4dde', '66ec052a5a08c7b9b35e29c7',
    '66ec09ec821e116aacb19620', '66f8c6b4bb02136c067c4480', '66efa5ea821e116aacb23268',
    '66f2ad89821e116aacb2ac92', '66f2a7a9821e116aacb2a721',
}


def load_longbench_v2_data(
    difficulty: str = None,
    length: str = None,
    max_tokens: int = None,
    critical_only: bool = False,
) -> List[Dict]:
    """
    Load LongBench v2 dataset from HuggingFace.

    The LongBench v2 dataset contains 503 long-context multiple-choice questions (4 options)
    across 6 domains: Single-Document QA, Multi-Document QA, Long In-context Learning,
    Long Structured Data Understanding, Code Repository Understanding, Long-dialogue History Understanding.

    Parameters
    ----------
    difficulty : str, optional
        Filter by difficulty level: 'easy' or 'hard'. If None, load all.
    length : str, optional
        Filter by context length category: 'short', 'medium', or 'long'. If None, load all.
    max_tokens : int, optional
        Maximum context length in tokens (counted with the Qwen3-4B tokenizer).
        If set, only include examples whose context is shorter than this threshold.
    critical_only : bool, optional
        If True, only include the 47 critical examples where original context helped
        and no-context failed (default: False).

    Returns
    -------
    data : list of dict
        List of articles in standardized format
    """
    import json
    from pathlib import Path

    _CACHE_PATH = Path(__file__).parent.parent / "data" / "longbenchv2_cache.jsonl"

    print(f"Loading LongBench v2...")
    if difficulty:
        print(f"  Filtering by difficulty: {difficulty}")
    if length:
        print(f"  Filtering by length: {length}")
    if max_tokens:
        print(f"  Filtering by max_tokens: {max_tokens:,}")
    if critical_only:
        print(f"  Filtering to {len(LONGBENCHV2_CRITICAL_IDS)} critical examples")

    if _CACHE_PATH.exists():
        print(f"  Using local cache: {_CACHE_PATH}")
        with open(_CACHE_PATH) as f:
            raw_entries = [json.loads(line) for line in f]
    else:
        from datasets import load_dataset as hf_load_dataset
        print(f"  Fetching from HuggingFace (run scripts/build_longbenchv2_cache.py to cache locally)")
        if max_tokens:
            from transformers import AutoTokenizer
            _tok_name = 'Qwen/Qwen3-4B'
            print(f"  Using tokenizer '{_tok_name}' for token counting")
            tokenizer = AutoTokenizer.from_pretrained(_tok_name)
            tokenizer.model_max_length = int(1e9)
        raw_entries = list(hf_load_dataset('THUDM/LongBench-v2', split='train'))
        if max_tokens:
            for entry in raw_entries:
                entry['num_tokens'] = len(tokenizer.encode(entry['context'], add_special_tokens=False))

    letter_to_idx = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    data = []
    skipped_by_length = 0

    for entry in raw_entries:
        # Apply filters
        if difficulty and entry['difficulty'] != difficulty:
            continue
        if length and entry['length'] != length:
            continue
        if max_tokens and entry.get('num_tokens', 0) > max_tokens:
            skipped_by_length += 1
            continue
        if critical_only and entry['_id'] not in LONGBENCHV2_CRITICAL_IDS:
            continue

        options = [entry['choice_A'], entry['choice_B'], entry['choice_C'], entry['choice_D']]
        gold_label = letter_to_idx.get(entry['answer'], 1)
        entry_id = entry['_id']

        article_entry = {
            'article_id': f"longbenchv2_{entry_id}",
            'title': f"{entry['domain']} / {entry['sub_domain']}",
            'article': entry['context'],
            'questions': [{
                'question': entry['question'],
                'options': options,
                'gold_label': gold_label,
                'question_unique_id': f"longbenchv2_{entry_id}",
            }],
        }
        data.append(article_entry)

    total_questions = sum(len(a['questions']) for a in data)
    print(f"Loaded {len(data)} articles with {total_questions} total questions")
    if skipped_by_length:
        print(f"  Skipped {skipped_by_length} articles exceeding {max_tokens:,} token limit")
    return data


def load_aime_data() -> List[Dict]:
    """
    Load AIME 2025 dataset from HuggingFace.

    The AIME 2025 dataset contains math competition problems with numeric answers (0-999).
    Loads both AIME I and AIME II (30 problems total).

    Returns
    -------
    data : list of dict
        List of problems in standardized format:
        - article_id: Unique identifier (e.g., 'aime2025_0')
        - title: Problem identifier (e.g., 'AIME 2025-I Problem 1')
        - article: The problem statement
        - questions: List with single entry containing:
            - question: Empty string (problem is in article)
            - ground_truth: The numeric answer (string, 0-999)
            - question_unique_id: Unique identifier
    """
    from datasets import load_dataset as hf_load_dataset

    print("Loading AIME 2025 dataset from HuggingFace...")

    data = []
    problem_idx = 0

    # Load both AIME I and AIME II
    for config_name in ['AIME2025-I', 'AIME2025-II']:
        dataset = hf_load_dataset('opencompass/AIME2025', config_name, split='test')

        for i, entry in enumerate(dataset):
            # Determine problem number (1-indexed within each competition)
            problem_num = i + 1
            if config_name == 'AIME2025-I':
                title = f"AIME 2025-I Problem {problem_num}"
            else:
                title = f"AIME 2025-II Problem {problem_num}"

            article_entry = {
                'article_id': f"aime2025_{problem_idx}",
                'title': title,
                'article': entry['question'],
                'questions': [{
                    'question': '',  # Problem is in article
                    'ground_truth': str(entry['answer']),
                    'question_unique_id': f"aime2025_{problem_idx}",
                }],
            }
            data.append(article_entry)
            problem_idx += 1

    print(f"Loaded {len(data)} AIME 2025 problems")
    return data


def load_ruler_data(
    context_length: int = 4096,
    task_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Load RULER benchmark dataset.

    Loads from the HuggingFace dataset ``simonjegou/ruler`` (pre-generated from the
    NVIDIA RULER benchmark). Each example becomes one "article" with one "question".

    The standardized format uses a special ``ruler_outputs`` field (list of reference
    answer strings) instead of ``options``/``gold_label``, and ``answer_prefix`` for
    guided generation. Scoring uses string-match rather than MCQ parsing.

    Parameters
    ----------
    context_length : int
        Target context length in tokens. Available on HuggingFace: 4096, 8192, 16384.
        For other lengths, pre-generate data with ``scripts/generate_ruler_data.py``
        and the loader will read from ``data/ruler/{context_length}/``.
    task_filter : str, optional
        If provided, only load examples whose ``task`` field matches this string
        (e.g., 'niah_single_1', 'vt', 'cwe', 'fwe', 'qa_1').

    Returns
    -------
    data : list of dict
        List of articles in standardized format with additional RULER-specific fields.
    """
    from datasets import load_dataset as hf_load_dataset

    config_name = str(context_length)
    hf_configs = {'4096', '8192', '16384'}

    # Try local pre-generated data first, then HuggingFace
    local_dir = Path(f'data/ruler/{config_name}')
    if local_dir.exists():
        # Load from local JSONL files
        print(f"Loading RULER {config_name} from local directory: {local_dir}")
        rows = []
        for jsonl_file in sorted(local_dir.glob('*.jsonl')):
            with open(jsonl_file) as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"No JSONL data found in {local_dir}")
    elif config_name in hf_configs:
        print(f"Loading RULER {config_name} from HuggingFace (simonjegou/ruler)...")
        dataset = hf_load_dataset('simonjegou/ruler', config_name, split='test')
        rows = list(dataset)
    else:
        raise ValueError(
            f"RULER context length {config_name} not available. "
            f"HuggingFace has: {sorted(hf_configs)}. "
            f"For other lengths, pre-generate data with: "
            f"python scripts/generate_ruler_data.py --context-length {context_length}"
        )

    # Apply task filter
    if task_filter:
        rows = [r for r in rows if r['task'] == task_filter]
        if not rows:
            available_tasks = sorted(set(r['task'] for r in rows))
            raise ValueError(f"No examples for task '{task_filter}'. Available: {available_tasks}")

    # Group by unique context to avoid redundant KV cache extraction.
    # Many RULER examples share the same context with different questions.
    context_groups = {}
    for idx, row in enumerate(rows):
        context = row['context']
        task = row['task']
        answer = row['answer']  # list of strings
        question = row['question']
        answer_prefix = row.get('answer_prefix', '')
        max_new_tokens = row.get('max_new_tokens', 128)

        # Use context hash as group key (contexts can be very long)
        ctx_key = hash(context)
        if ctx_key not in context_groups:
            context_groups[ctx_key] = {
                'context': context,
                'questions': [],
                'article_idx': len(context_groups),
            }

        context_groups[ctx_key]['questions'].append({
            'question': question,
            'ruler_outputs': answer,
            'answer_prefix': answer_prefix,
            'max_new_tokens': max_new_tokens,
            'task': task,
            'question_unique_id': f"ruler_{config_name}_{task}_{idx}",
        })

    # Convert to standardized format
    data = []
    for ctx_key, group in context_groups.items():
        article_entry = {
            'article_id': f"ruler_{config_name}_{group['article_idx']}",
            'title': f"RULER {config_name} (article {group['article_idx']})",
            'article': group['context'],
            'questions': group['questions'],
        }
        data.append(article_entry)

    tasks_found = sorted(set(q['task'] for a in data for q in a['questions']))
    total_questions = sum(len(a['questions']) for a in data)
    print(f"Loaded {len(data)} articles with {total_questions} total questions")
    print(f"  Tasks: {tasks_found}")
    return data


def load_qasper_data() -> List[Dict]:
    """
    Load QASPER dataset from HuggingFace.

    The QASPER dataset contains NLP research papers with information-seeking questions.
    Each question has multiple annotator answers that can be extractive, abstractive
    (free-form), yes/no, or unanswerable.

    Uses the **validation** split (281 papers, ~1005 questions) since the test split
    has no answers.

    Returns
    -------
    data : list of dict
        List of papers in standardized format:
        - article_id: Unique identifier (e.g., 'qasper_1234.5678')
        - title: Paper title
        - article: Abstract + full text (sections concatenated)
        - questions: List of questions, each with:
            - question: Question text
            - qasper_answers: List of reference answer strings (one per annotator)
            - question_unique_id: Unique identifier
    """
    import io
    import tarfile
    import urllib.request

    print("Loading QASPER dataset (validation split)...")

    # Download from S3 (same source as the HuggingFace dataset script)
    tgz_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    cache_dir = Path('data/qasper_cache')
    cache_file = cache_dir / 'qasper-dev-v0.3.json'

    if not cache_file.exists():
        print(f"  Downloading from {tgz_url}...")
        cache_dir.mkdir(parents=True, exist_ok=True)
        response = urllib.request.urlopen(tgz_url)
        tgz_bytes = io.BytesIO(response.read())
        with tarfile.open(fileobj=tgz_bytes, mode='r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('qasper-dev-v0.3.json'):
                    f = tar.extractfile(member)
                    cache_file.write_bytes(f.read())
                    break
        print(f"  Cached to {cache_file}")

    with open(cache_file, 'r') as f:
        raw_data = json.load(f)  # {paper_id: {paper_data}}

    print(f"  Loaded {len(raw_data)} papers from dev split")

    data = []
    total_questions = 0
    skipped_questions = 0

    for paper_id, paper in raw_data.items():
        title = paper['title']
        abstract = paper['abstract']

        # Reconstruct full text from sections
        full_text_parts = []
        for section in paper['full_text']:
            section_name = section['section_name']
            paragraphs = section['paragraphs']
            if section_name:
                full_text_parts.append(f"## {section_name}")
            full_text_parts.extend(paragraphs)

        full_text = "\n\n".join(full_text_parts)
        article_text = f"{abstract}\n\n{full_text}".strip()

        # Process questions
        questions = []
        for qa in paper['qas']:
            question_text = qa['question']
            question_id = qa['question_id']

            # Collect reference answers from all annotators
            reference_answers = []
            for annotator_answer in qa['answers']:
                ans = annotator_answer['answer']

                if ans['unanswerable']:
                    reference_answers.append("Unanswerable")
                elif ans['yes_no'] is not None:
                    reference_answers.append("Yes" if ans['yes_no'] else "No")
                elif ans['extractive_spans'] and len(ans['extractive_spans']) > 0:
                    reference_answers.append(", ".join(ans['extractive_spans']))
                elif ans['free_form_answer'] and ans['free_form_answer'].strip():
                    reference_answers.append(ans['free_form_answer'].strip())
                # else: skip empty answers

            if not reference_answers:
                skipped_questions += 1
                continue

            questions.append({
                'question': question_text,
                'qasper_answers': reference_answers,
                'question_unique_id': f"qasper_{paper_id}_{question_id}",
            })
            total_questions += 1

        if not questions:
            continue

        article_entry = {
            'article_id': f"qasper_{paper_id}",
            'title': title,
            'article': article_text,
            'questions': questions,
        }
        data.append(article_entry)

    print(f"Loaded {len(data)} papers with {total_questions} total questions")
    if skipped_questions:
        print(f"  Skipped {skipped_questions} questions with no valid answers")
    return data


# Registry of available dataset loaders
DATASET_LOADERS = {
    'quality': load_quality_data,
    'longhealth': load_longhealth_data,
    'longcodeqa': load_longcodeqa_data,
    'longsweb': load_longsweb_data,
    'aime2025': load_aime_data,
    'longbenchv2': load_longbench_v2_data,
    'ruler': load_ruler_data,
    'qasper': load_qasper_data,
}

# Default dataset paths
DATASET_PATHS = {
    'quality': 'data/QuALITY.v1.0.1.htmlstripped.dev',
    'longhealth': 'data/longhealth_benchmark_v5.json',
}


def load_dataset(dataset_name: str, include_diagnosis: bool = True) -> List[Dict]:
    """
    Load a dataset by name.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset. Supported formats:
        - 'quality': Load QuALITY dataset
        - 'longhealth': Load all LongHealth patients (each as separate article)
        - 'longhealthX': Group patients into articles with X patients per article
          (e.g., 'longhealth10' groups 10 patients per article, creates floor(n/10) articles)
        - 'lqa32k', 'lqa64k', 'lqa128k', 'lqa256k', 'lqa512k', 'lqa1m': Load LongCodeQA
          at specified context length (downloaded from HuggingFace)
        - 'longsweb' or 'longswebXXk': Load LongSWE-bench dataset for perplexity evaluation
          on patches. Options: 'longsweb' (default 64K), 'longsweb32k', 'longsweb64k', 'longsweb128k'
        - 'longbenchv2': Load LongBench v2 (503 long-context MCQ examples)
        - 'longbenchv2_easy'/'longbenchv2_hard': Filter by difficulty
        - 'longbenchv2_short'/'longbenchv2_medium'/'longbenchv2_long': Filter by length category
        - 'longbenchv2_100k': Only contexts under ~100k tokens (similarly '32k', '64k', etc.)
        - 'longbenchv2_critical': Only the 47 critical examples where context is essential
        - 'longbenchv2_100k_critical': Critical examples filtered to under 100k tokens
        - 'qasper': Load QASPER dataset (NLP papers with free-form QA, validation split)
    include_diagnosis : bool
        Whether to include diagnosis in patient info (default: True)

    Returns
    -------
    data : list of dict
        Loaded dataset in standardized format

    Raises
    ------
    ValueError
        If dataset_name format is not recognized

    Examples
    --------
    >>> # Load QuALITY dataset
    >>> data = load_dataset('quality')

    >>> # Load all LongHealth patients (each as separate article)
    >>> data = load_dataset('longhealth')

    >>> # Group 10 patients per article
    >>> data = load_dataset('longhealth10')

    >>> # Load LongCodeQA at 32K context length
    >>> data = load_dataset('lqa32k')

    >>> # Load LongCodeQA at 128K context length
    >>> data = load_dataset('lqa128k')

    >>> # Load LongSWE-bench for perplexity evaluation (default 64K)
    >>> data = load_dataset('longsweb')

    >>> # Load LongSWE-bench at 128K context length
    >>> data = load_dataset('longsweb128k')
    """
    if dataset_name == 'quality':
        data_path = DATASET_PATHS['quality']
        return load_quality_data(data_path)

    elif dataset_name.startswith('longhealth'):
        data_path = DATASET_PATHS['longhealth']

        # Parse the number of patients per article from the dataset name
        patients_per_article = 1 if dataset_name == 'longhealth' else int(dataset_name[len('longhealth'):])

        # Load ALL patients grouped by patients_per_article
        return load_longhealth_data(
            data_path,
            include_diagnosis=include_diagnosis,
            patients_per_article=patients_per_article
        )

    elif dataset_name.startswith('lqa'):
        # Parse context length from dataset name (e.g., 'lqa32k' -> '32K')
        context_suffix = dataset_name[3:].upper()  # '32k' -> '32K'
        return load_longcodeqa_data(context_length=context_suffix)

    elif dataset_name.startswith('longsweb'):
        # Parse context length from dataset name (e.g., 'longsweb64k' -> '64K')
        # Default to 64K if no suffix provided
        if dataset_name == 'longsweb':
            context_suffix = '64K'
        else:
            context_suffix = dataset_name[len('longsweb'):].upper()  # 'longsweb64k' -> '64K'
        return load_longsweb_data(context_length=context_suffix)

    elif dataset_name == 'aime2025':
        return load_aime_data()

    elif dataset_name.startswith('ruler'):
        # Parse context length and optional task filter from dataset name
        # Formats: 'ruler_4k', 'ruler_128k', 'ruler_4k_niah_single_1'
        suffix = dataset_name[len('ruler'):]
        if not suffix or not suffix.startswith('_'):
            raise ValueError(
                f"RULER dataset name must include context length, e.g. 'ruler_4k', 'ruler_128k'. "
                f"Optionally filter by task: 'ruler_4k_niah_single_1'"
            )
        suffix = suffix[1:]  # strip leading '_'

        # Parse context length (first token before optional task filter)
        # Context length is like '4k', '8k', '16k', '32k', '64k', '128k'
        match = re.match(r'^(\d+k)', suffix)
        if not match:
            raise ValueError(
                f"Could not parse RULER context length from '{dataset_name}'. "
                f"Expected format: 'ruler_4k', 'ruler_128k', etc."
            )
        context_str = match.group(1)
        context_length = int(context_str[:-1]) * 1024  # '4k' -> 4096
        task_filter = suffix[len(context_str):]
        if task_filter:
            task_filter = task_filter.lstrip('_')
        else:
            task_filter = None
        return load_ruler_data(context_length=context_length, task_filter=task_filter)

    elif dataset_name == 'qasper':
        return load_qasper_data()

    elif dataset_name.startswith('longbenchv2'):
        # Parse optional filters from dataset name
        # Formats: 'longbenchv2', 'longbenchv2_easy', 'longbenchv2_hard',
        #          'longbenchv2_short', 'longbenchv2_medium', 'longbenchv2_long',
        #          'longbenchv2_100k' (max 100k tokens)
        suffix = dataset_name[len('longbenchv2'):]
        difficulty = None
        length = None
        max_tokens = None
        critical_only = False
        if suffix:
            suffix = suffix.lstrip('_')
            if suffix == '100k_critical':
                max_tokens = 100_000
                critical_only = True
            elif suffix == 'critical':
                critical_only = True
            elif suffix in ('easy', 'hard'):
                difficulty = suffix
            elif suffix in ('short', 'medium', 'long'):
                length = suffix
            elif re.match(r'^\d+k$', suffix):
                max_tokens = int(suffix[:-1]) * 1000
            else:
                raise ValueError(
                    f"Unknown LongBench v2 filter: '{suffix}'. "
                    f"Use 'easy'/'hard' for difficulty, 'short'/'medium'/'long' for length, "
                    f"'Nk' (e.g., '100k') for max token limit, or 'critical'/'100k_critical'."
                )
        return load_longbench_v2_data(difficulty=difficulty, length=length, max_tokens=max_tokens, critical_only=critical_only)

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported formats: 'quality', 'longhealth', 'longhealthX' (e.g., 'longhealth10'), "
            f"'lqaXX' (e.g., 'lqa32k', 'lqa128k', 'lqa1m'), 'longsweb' or 'longswebXXk' (e.g., 'longsweb64k', 'longsweb128k'), "
            f"'aime2025', 'longbenchv2' (or 'longbenchv2_easy', 'longbenchv2_hard', 'longbenchv2_short', 'longbenchv2_100k', 'longbenchv2_critical', 'longbenchv2_100k_critical', etc.), "
            f"'ruler_Xk' (e.g., 'ruler_4k', 'ruler_128k', 'ruler_4k_niah_single_1'), "
            f"'qasper'"
        )


# Datasets that use ground_truth perplexity evaluation instead of multiple-choice QA
# Use prefixes for datasets that have variants (e.g., longsweb64k, longsweb128k)
PERPLEXITY_DATASET_PREFIXES = {'longsweb'}


def is_perplexity_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset uses perplexity-based evaluation (ground_truth) instead of multiple-choice QA.

    Perplexity datasets have questions with 'ground_truth' field instead of 'options' and 'gold_label'.
    For these datasets, we compute the perplexity of the ground truth text given the context,
    rather than generating answers and checking correctness.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if the dataset uses perplexity-based evaluation
    """
    return any(dataset_name.startswith(prefix) for prefix in PERPLEXITY_DATASET_PREFIXES)


# Datasets that use string-match evaluation (RULER benchmark)
RULER_DATASET_PREFIX = 'ruler'


def is_ruler_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset uses RULER-style string-match evaluation.

    RULER datasets have questions with 'ruler_outputs' (list of reference answer strings)
    instead of 'options'/'gold_label'. Evaluation uses substring matching rather than
    MCQ parsing.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if the dataset uses RULER-style evaluation
    """
    return dataset_name.startswith(RULER_DATASET_PREFIX)


# Datasets that use token F1 evaluation (QASPER benchmark)
QASPER_DATASET_PREFIX = 'qasper'


def is_qasper_dataset(dataset_name: str) -> bool:
    """
    Check if a dataset uses QASPER-style token F1 evaluation.

    QASPER datasets have questions with 'qasper_answers' (list of reference answer strings
    from multiple annotators) instead of 'options'/'gold_label'. Evaluation uses token-level
    F1 (SQuAD-style) with max F1 across annotators.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset

    Returns
    -------
    bool
        True if the dataset uses QASPER-style evaluation
    """
    return dataset_name.startswith(QASPER_DATASET_PREFIX)
