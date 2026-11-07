SHELL := /bin/bash

DATA_DIR ?= new-splitted-data
DATASET ?= failed_test

REFERENCE := $(DATASET)

LLM_TASKS := \
	     simple \
	     localization \
	     one_shot \
	     five_shot \
	     five_shot_localization \

LLM_MODELS := \
	DeepSeek_R1_Distill_Qwen_7B-reasoning \
	gemma_7b_it-noreasoning \
	gemma_3_12b_it-noreasoning \
	gpt_oss_20b-low-reasoning \
	gpt_oss_20b-medium-reasoning \
	Llama_3.1_8B_Instruct-noreasoning \
	Qwen3_8B-noreasoning \
	Qwen3_8B-reasoning \
	phi_4-noreasoning \

LLM_METHODS := sft_llm_Qwen3_8B-reasoning $(foreach llm_task, $(LLM_TASKS), \
	$(foreach llm_model, $(LLM_MODELS), \
		$(llm_task)_llm_$(llm_model) \
	) \
)

CLOSED_LLM_METHODS := $(foreach llm_task, $(LLM_TASKS), \
	$(llm_task)_closed_llm_gpt_5 \
)
METHODS ?= $(LLM_METHODS) $(CLOSED_LLM_METHODS)

.DELETE_ON_ERROR:

.SECONDARY:

.PHONY: default
default: all

###############################################################################
# XXX: LLM-related repaired results. Note that prerequisites are treated
# manually
# TODO: Use for-each statement
###############################################################################
$(DATA_DIR)/repaired-patterns/simple_llm_%.txt: \
		$(DATA_DIR)/large-language-models/simple/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

$(DATA_DIR)/repaired-patterns/localization_llm_%.txt: \
		$(DATA_DIR)/large-language-models/localization/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

$(DATA_DIR)/repaired-patterns/one_shot_llm_%.txt: \
		$(DATA_DIR)/large-language-models/one_shot/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

$(DATA_DIR)/repaired-patterns/five_shot_llm_%.txt: \
		$(DATA_DIR)/large-language-models/five_shot/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

$(DATA_DIR)/repaired-patterns/sft_llm_%.txt: \
		$(DATA_DIR)/large-language-models/sft/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

$(DATA_DIR)/repaired-patterns/five_shot_localization_llm_%.txt: \
		$(DATA_DIR)/large-language-models/five_shot_localization/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_repaired_patterns_from_llm.py < $< > $@

#############################################################################
$(DATA_DIR)/repaired-patterns/simple_closed_llm_%.txt: \
		$(DATA_DIR)/closed-llm/simple/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format closed_llm < $< > $@

$(DATA_DIR)/repaired-patterns/localization_closed_llm_%.txt: \
		$(DATA_DIR)/closed-llm/localization/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format closed_llm < $< > $@

$(DATA_DIR)/repaired-patterns/one_shot_closed_llm_%.txt: \
		$(DATA_DIR)/closed-llm/one_shot/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format closed_llm < $< > $@

$(DATA_DIR)/repaired-patterns/five_shot_closed_llm_%.txt: \
		$(DATA_DIR)/closed-llm/five_shot/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format closed_llm < $< > $@

$(DATA_DIR)/repaired-patterns/five_shot_localization_closed_llm_%.txt: \
		$(DATA_DIR)/closed-llm/five_shot_localization/%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format closed_llm < $< > $@
#############################################################################


###############################################################################
# Without any repair
###############################################################################
$(DATA_DIR)/repaired-patterns/no_repair-%.txt: $(DATA_DIR)/patterns/%.txt
	@mkdir -p $(dir $@)
	cp -f $< $@
###############################################################################


###############################################################################
# Targets for `scripts/print_vulnerability_ratio.sh`
###############################################################################
VULNERABILITY_TARGET := $(foreach method,$(METHODS),$(addprefix $(DATA_DIR)/,\
	results/redos_hunter-$(method)-$(DATASET).jsonl \
	results/redos_hunter-$(method)-$(DATASET).jsonl \
))
VULNERABILITY_TARGET += $(DATA_DIR)/results/redos_hunter-$(DATASET).jsonl

.PHONY: print-vulnerability-target
print-vulnerability-target:
	@echo $(VULNERABILITY_TARGET)

.PHONY: vulnerability-target
vulnerability-target: $(VULNERABILITY_TARGET)
###############################################################################


###############################################################################
# Targets for `scripts/print_edit_distance.sh`
###############################################################################
EDIT_DISTANCE_TARGET := $(foreach method,$(METHODS),$(addprefix $(DATA_DIR)/,\
	repaired-patterns/$(method)-$(DATASET).txt \
))
EDIT_DISTANCE_TARGET += patterns/$(DATASET).txt


.PHONY: print-edit-distance-target
print-edit-distance-target:
	@echo $(EDIT_DISTANCE_TARGET)

.PHONY: edit-distance-target
edit-distance-target: $(EDIT_DISTANCE_TARGET)
###############################################################################


###############################################################################
# Targets for `scripts/print_jaccard_similarity.sh`
###############################################################################
JACCARD_SIMILARITY_TARGET := $(foreach method,$(METHODS),$(addprefix $(DATA_DIR)/,\
	full-matching/reference-test-cases/xeger-$(method)-$(DATASET).jsonl \
	full-matching/reference-patterns/xeger-$(method)-$(DATASET).jsonl \
))

.PHONY: print-jaccard-similarity-target
print-jaccard-similarity-target:
	@echo $(JACCARD_SIMILARITY_TARGET)

.PHONY: jaccard-similarity-target
jaccard-similarity-target: $(JACCARD_SIMILARITY_TARGET)
###############################################################################

TARGET := \
	  $(VULNERABILITY_TARGET) \
	  $(EDIT_DISTANCE_TARGET) \
	  $(JACCARD_SIMILARITY_TARGET)

###############################################################################
# Basics
###############################################################################
$(DATA_DIR)/patterns/sl_regexes.txt: raw-data/sl_regexes.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py < $< > $@

$(DATA_DIR)/patterns/snort3.txt: raw-data/snort3.txt
	@mkdir -p $(dir $@)
	cp -f $< $@

$(DATA_DIR)/patterns/filtered_%.txt: $(DATA_DIR)/results/redos_hunter-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/filter_vulnerable_patterns.py < $< >$@
###############################################################################


###############################################################################
# Regex Scalpel
###############################################################################
$(DATA_DIR)/raw-outputs/regex_scalpel-%.jsonl: $(DATA_DIR)/patterns/%.txt
	@mkdir -p $(dir $@)
	python scripts/run_regex_scalpel.py < $< > $@

$(DATA_DIR)/results/regex_scalpel-%.jsonl: \
		$(DATA_DIR)/raw-outputs/regex_scalpel-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/parse_regex_scalpel.py < $< > $@

$(DATA_DIR)/repaired-patterns/regex_scalpel-%.txt: \
		$(DATA_DIR)/results/regex_scalpel-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_patterns_from_jsonl.py \
		--format regex_scalpel < $< > $@
###############################################################################


###############################################################################
# Remedy
# XXX: Raw-outputs should be obtained manually
###############################################################################
$(DATA_DIR)/repaired-patterns/remedy-%.txt: \
		$(DATA_DIR)/results/remedy-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/parse_remedy.py < $< > $@
###############################################################################


###############################################################################
# Redos Hunter
###############################################################################
$(DATA_DIR)/raw-outputs/redos_hunter-%.jsonl: $(DATA_DIR)/patterns/%.txt
	@mkdir -p $(dir $@)
	python scripts/run_redos_hunter.py < $< > $@

$(DATA_DIR)/raw-outputs/redos_hunter-%.jsonl: \
		$(DATA_DIR)/repaired-patterns/%.txt
	@mkdir -p $(dir $@)
	python scripts/run_redos_hunter.py < $< > $@

$(DATA_DIR)/results/redos_hunter-%.jsonl: \
		$(DATA_DIR)/raw-outputs/redos_hunter-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/parse_redos_hunter.py < $< > $@
###############################################################################


###############################################################################
# Xeger
###############################################################################
$(DATA_DIR)/test-cases/full/xeger-%.jsonl: $(DATA_DIR)/patterns/%.txt
	@mkdir -p $(dir $@)
	python scripts/run_xeger.py < $< > $@

$(DATA_DIR)/test-cases/full/xeger-%.jsonl: $(DATA_DIR)/repaired-patterns/%.txt
	@mkdir -p $(dir $@)
	python scripts/run_xeger.py < $< > $@
###############################################################################


###############################################################################
# Matching
###############################################################################
$(DATA_DIR)/full-matching/reference-test-cases/xeger-%.jsonl: \
		$(DATA_DIR)/repaired-patterns/%.txt \
		$(DATA_DIR)/test-cases/full/xeger-$(REFERENCE).jsonl
	@mkdir -p $(dir $@)
	python scripts/get_mathing_results.py $+ > $@

$(DATA_DIR)/full-matching/reference-patterns/xeger-%.jsonl: \
		$(DATA_DIR)/patterns/$(REFERENCE).txt \
		$(DATA_DIR)/test-cases/full/xeger-%.jsonl
	@mkdir -p $(dir $@)
	python scripts/get_mathing_results.py $+ > $@
###############################################################################


.PHONY: all
all: $(TARGET)
