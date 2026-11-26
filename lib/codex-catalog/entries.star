"""Codex's catalog entries, the models that `codex_entry` has one for.

A model's entry is in a file of its own, with its provenance; add a model by its file here and in
`_ENTRIES`.
"""

load("//lib/codex-catalog/glm_5_3.star", GLM_5_3 = "ENTRY")
load("//lib/codex-catalog/gpt_6_1_terra.star", GPT_6_1_TERRA = "ENTRY")
load("//lib/codex-catalog/gpt_6_2_astra.star", GPT_6_2_ASTRA = "ENTRY")
load("//lib/codex-catalog/gpt_6_luna.star", GPT_6_LUNA = "ENTRY")

# Each entry's JSON text, by the `slug` it holds, which is the model's name.
_ENTRIES = {
    "glm-5.3": GLM_5_3,
    "gpt-6.1-terra": GPT_6_1_TERRA,
    "gpt-6.2-astra": GPT_6_2_ASTRA,
    "gpt-6-luna": GPT_6_LUNA,
}

def codex_entry(model):
    """Codex's catalog entry for `model`, which `codex()` takes as its `catalog_entry`.

    Args:
      model: the model's name, or `None` for an agent other than Codex.

    Returns:
      The entry, a dict; or `None` if there is none or `model` is `None`.
    """
    if model == None:
        return None
    text = _ENTRIES.get(model)
    if text == None:
        return None
    return json.decode(text)
