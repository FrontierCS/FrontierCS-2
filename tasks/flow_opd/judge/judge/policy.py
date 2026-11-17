import ast
from pathlib import Path

SYMBOLS = {
 'flow_grpo/diffusers_patch/sd3_pipeline_with_logprob.py':'pipeline_with_logprob',
 'flow_grpo/diffusers_patch/sd3_sde_with_logprob.py':'sde_step_with_logprob',
}


def skeleton(path, symbol):
    tree = ast.parse(Path(path).read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==symbol]
    if len(functions)!=1:
        raise ValueError('Required entry point missing or duplicated.')
    functions[0].body = [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def check_surface(workdir, pristine):
    for path,symbol in SYMBOLS.items():
        if skeleton(Path(workdir)/path,symbol)!=skeleton(Path(pristine)/path,symbol):
            raise ValueError('Only the two computational bodies may change; retain imports and signatures.')
