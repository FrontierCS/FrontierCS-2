simple_methods="\
  simple_llm_Qwen3_8B-noreasoning \
  simple_llm_Llama_3.1_8B_Instruct-noreasoning \
  simple_llm_gemma_3_12b_it-noreasoning \
  simple_llm_phi_4-noreasoning \
  simple_llm_Qwen3_8B-reasoning \
  simple_llm_DeepSeek_R1_Distill_Qwen_7B-reasoning \
  simple_llm_gpt_oss_20b-low-reasoning \
  simple_llm_gpt_oss_20b-medium-reasoning \
  simple_closed_llm_gpt_5 \
"

localization_methods="\
  localization_llm_Qwen3_8B-noreasoning \
  localization_llm_Llama_3.1_8B_Instruct-noreasoning \
  localization_llm_gemma_3_12b_it-noreasoning \
  localization_llm_phi_4-noreasoning \
  localization_llm_Qwen3_8B-reasoning \
  localization_llm_DeepSeek_R1_Distill_Qwen_7B-reasoning \
  localization_llm_gpt_oss_20b-low-reasoning \
  localization_llm_gpt_oss_20b-medium-reasoning \
  localization_closed_llm_gpt_5 \
"

one_shot_methods="\
  one_shot_llm_Qwen3_8B-noreasoning \
  one_shot_llm_Llama_3.1_8B_Instruct-noreasoning \
  one_shot_llm_gemma_3_12b_it-noreasoning \
  one_shot_llm_phi_4-noreasoning \
  one_shot_llm_Qwen3_8B-reasoning \
  one_shot_llm_DeepSeek_R1_Distill_Qwen_7B-reasoning \
  one_shot_llm_gpt_oss_20b-low-reasoning \
  one_shot_llm_gpt_oss_20b-medium-reasoning \
  one_shot_closed_llm_gpt_5 \
"
five_shot_methods="\
  five_shot_llm_Qwen3_8B-noreasoning \
  five_shot_llm_Llama_3.1_8B_Instruct-noreasoning \
  five_shot_llm_gemma_3_12b_it-noreasoning \
  five_shot_llm_phi_4-noreasoning \
  five_shot_llm_Qwen3_8B-reasoning \
  five_shot_llm_DeepSeek_R1_Distill_Qwen_7B-reasoning \
  five_shot_llm_gpt_oss_20b-low-reasoning \
  five_shot_llm_gpt_oss_20b-medium-reasoning \
  five_shot_closed_llm_gpt_5 \
"

five_shot_localization_methods="\
  five_shot_localization_llm_Qwen3_8B-noreasoning \
  five_shot_localization_llm_Llama_3.1_8B_Instruct-noreasoning \
  five_shot_localization_llm_gemma_3_12b_it-noreasoning \
  five_shot_localization_llm_phi_4-noreasoning \
  five_shot_localization_llm_Qwen3_8B-reasoning \
  five_shot_localization_llm_DeepSeek_R1_Distill_Qwen_7B-reasoning \
  five_shot_localization_llm_gpt_oss_20b-low-reasoning \
  five_shot_localization_llm_gpt_oss_20b-medium-reasoning \
  five_shot_localization_closed_llm_gpt_5 \
"

methods="\
  regex_scalpel \
  ${simple_methods} \
  ${localization_methods} \
  ${one_shot_methods} \
  ${five_shot_methods} \
  ${five_shot_localization_methods}
"
