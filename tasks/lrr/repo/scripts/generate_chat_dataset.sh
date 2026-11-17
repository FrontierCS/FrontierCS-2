mkdir -p new-splitted-data/training

python scripts/scalpel_to_chat.py \
  < new-splitted-data/results/regex_scalpel-succeeded_train.jsonl \
  > new-splitted-data/training/chat_train.jsonl

