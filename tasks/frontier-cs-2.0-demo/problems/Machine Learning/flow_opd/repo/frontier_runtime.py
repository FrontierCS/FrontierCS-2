"""Task-owned runtime adapter; source-only draft, not GPU-qualified.

Executes the retained shell launcher and trainer. All scientific sampling,
reward, loss, backward and optimizer operations remain in upstream/solver code.
Explicit profiles are required; this file contains no workload identities or
reference implementation. Runtime hooks are instrumentation, not a sandbox.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.resources
import inspect
import resource
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types

ENTRYPOINT = 'scripts/single_node/sd3_opd_example.sh'
TRAINER = 'scripts/train_sd3.py'
LAUNCH_ARGS = ['launch','--config_file','scripts/accelerate_configs/multi_gpu.yaml',
               '--num_processes=8','--main_process_port','29501',TRAINER,
               '--config','config/grpo.py:general_ocr_sd3_8gpu_opd']


class Completed(BaseException):
    pass


def load_transition_guard(profile):
    if profile['qualification_status'] != 'qualified':
        return None
    tolerance = profile.get('transition_tolerance')
    if (not isinstance(tolerance, dict) or set(tolerance) != {'rtol', 'atol'}
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in tolerance.values())):
        raise ValueError('Qualified profiles require frozen numerical tolerances.')
    path = Path(__file__).resolve().parents[2]/'judge/transition_checks.py'
    spec = importlib.util.spec_from_file_location('_frontier_transition_guard', path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard.check_transition


def read_profile(path):
    profile = json.loads(Path(path).read_text())
    if profile.get('qualification_status') not in {'qualified', 'calibration'}:
        raise ValueError('An authorized, qualified run profile is required.')
    for key in ('updates','train_batch_size','repeat_count','batches_per_epoch',
                'gradient_accumulation_steps','train_microbatch_size','evaluation_batch_size',
                'max_runner_seconds'):
        if type(profile.get(key)) is not int or profile[key]<=0:
            raise ValueError('Positive qualified resource settings required.')
    if profile['train_batch_size'] % profile['repeat_count']:
        raise ValueError('One-process repeated rollout groups do not divide the batch.')
    if profile['repeat_count']<2 or profile['updates']%8:
        raise ValueError('Qualified repeated sampling and complete EMA boundaries required.')
    if not profile.get('evaluation') or any('seed' not in x or 'prompt' not in x for x in profile['evaluation']):
        raise ValueError('Paired evaluation randomness required.')
    return profile


def transform_trainer(source):
    """Small explicit AST runtime adapters; retain original file bytes on disk."""
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    counts = {'configure':0,'optimizer':0,'bind':0,'periodic':0,'unwrap':0,'boundary':0}

    def hook(statement):
        return ast.parse(statement).body[0]

    class Adapter(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if any(isinstance(t,ast.Name) and t.id=='config' for t in node.targets) and isinstance(node.value,ast.Attribute) and ast.unparse(node.value)=='FLAGS.config':
                counts['configure']+=1
                return [node,hook('_frontier.configure(config)')]
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='optimizer_cls':
                counts['optimizer']+=1
                return [node,hook('_frontier.observe_optimizer(optimizer, transformer_trainable_parameters)')]
            if ast.unparse(node.value)=='0' and any(isinstance(t,ast.Name) and t.id=='global_step' for t in node.targets):
                counts['bind']+=1
                return [node,hook('_frontier.bind(locals(), globals())')]
            return node

        def visit_If(self,node):
            # Suppress periodic held-out selection; evaluate/save prescribed endpoint only.
            condition=ast.unparse(node.test)
            if condition.startswith('epoch % config.eval_freq') or condition.startswith('epoch % config.save_freq'):
                counts['periodic']+=1
                return ast.copy_location(ast.Pass(),node)
            return self.generic_visit(node)

        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='transformer' and node.attr=='module':
                counts['unwrap']+=1
                return ast.copy_location(ast.parse('accelerator.unwrap_model(transformer)',mode='eval').body,node)
            return self.generic_visit(node)

        def visit_AugAssign(self,node):
            if isinstance(node.target,ast.Name) and node.target.id=='epoch':
                counts['boundary']+=1
                return [node,hook('_frontier.boundary(locals(), globals())')]
            return node
    Adapter().visit(main)
    expected={'configure':1,'optimizer':1,'bind':1,'periodic':2,'unwrap':1,'boundary':1}
    if counts!=expected:
        raise ValueError('Retained trainer no longer matches the reviewed runtime adapter.')
    return ast.fix_missing_locations(tree),counts


class Runtime:
    def __init__(self, profile, output):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        import torch
        self.torch=torch
        # r008 matched consumed inputs/losses but diverged in accumulated gradients.
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        if not torch.cuda.is_available():
            raise RuntimeError('A Driver NVIDIA GPU runner is required; no CPU fallback.')
        self.profile=profile; self.output=Path(output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        self.start=time.monotonic(); self.updates=0; self.loss_calls=0
        self.rollouts=0; self.teacher_forwards=0; self.student_forwards=0
        self.base_forwards=0; self.evaluation=False; self.eval_index=0
        self.rewards=[]; self.eval_latent_hashes=[]
        self.result={'complete':False,'verified_optimizer_updates':0,
                     'backend':{'torch_version':torch.__version__,
                                'cuda_version':torch.version.cuda,
                                'device_name':torch.cuda.get_device_name(),
                                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                                'cudnn_benchmark':torch.backends.cudnn.benchmark,
                                'cudnn_deterministic':torch.backends.cudnn.deterministic,
                                'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']}}
        self.install_offline()

    def install_offline(self):
        # Python 3.12 removed imp. Only the upstream config's load_source is used.
        if importlib.util.find_spec('imp') is None:
            module=types.ModuleType('imp')
            def load_source(name,path):
                spec=importlib.util.spec_from_file_location(name,path)
                result=importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
                return result
            module.load_source=load_source; sys.modules['imp']=module
        # prompts.py eagerly requests a nonexistent unused package. Resolve its
        # resource directory without changing any prompt loader computation.
        original_files=importlib.resources.files
        def files(anchor=None):
            if anchor=='flow_grpo.assets':
                return Path.cwd()/'flow_grpo/assets'
            return original_files(anchor)
        importlib.resources.files=files
        # Keep the original scorer; inject only offline model directories.
        from paddleocr import PaddleOCR
        import paddleocr
        model_root=Path(os.environ['FRONTIER_MODEL_ROOT'])
        ocr_root=model_root/'ocr'
        for name in ('en_PP-OCRv3_det_infer','en_PP-OCRv4_rec_infer','ch_ppocr_mobile_v2.0_cls_infer'):
            if not (ocr_root/name/'inference.pdmodel').is_file():
                raise RuntimeError('Offline OCR assets are not staged in this runner.')
        def offline_ocr(*args,**kwargs):
            kwargs.update(det_model_dir=str(ocr_root/'en_PP-OCRv3_det_infer'),
                          rec_model_dir=str(ocr_root/'en_PP-OCRv4_rec_infer'),
                          cls_model_dir=str(ocr_root/'ch_ppocr_mobile_v2.0_cls_infer'))
            return PaddleOCR(*args,**kwargs)
        paddleocr.PaddleOCR=offline_ocr
        # Private runs never send W&B logs or image captions over the network.
        import wandb
        wandb.init=lambda *args,**kwargs: None
        wandb.log=lambda *args,**kwargs: None
        wandb.Image=lambda *args,**kwargs: None
        from accelerate import Accelerator
        backward=Accelerator.backward
        runtime=self
        def audited_backward(accelerator,loss,*args,**kwargs):
            if not runtime.torch.isfinite(loss).all():
                raise ValueError('Nonfinite consumed loss.')
            caller=sys._getframe(1).f_locals
            if getattr(runtime,'updates',None)==0 and 'sample' in caller:
                sample=caller['sample']; j=caller['j']
                consumed=[sample[key][:,j] for key in
                          ('latents','next_latents','log_probs','advantages','timesteps')]
                consumed.extend(caller[key] for key in ('embeds','pooled_embeds','mask','perm'))
                runtime.result.setdefault('first_update_inputs',[]).append({
                    'timestep_index':j,'consumed':runtime.tensor_digest(consumed),
                    'cpu_rng':runtime.tensor_digest([runtime.torch.get_rng_state()]),
                    'cuda_rng':runtime.tensor_digest([runtime.torch.cuda.get_rng_state()])})
            runtime.result.setdefault('loss_sha256',[]).append(runtime.tensor_digest([loss.reshape(1)]))
            runtime.loss_calls+=1
            return backward(accelerator,loss,*args,**kwargs)
        Accelerator.backward=audited_backward
        import flow_grpo.rewards
        original_multi_score=flow_grpo.rewards.multi_score
        scorer_count=0
        def multi_score(*args,**kwargs):
            nonlocal scorer_count
            scorer_index=scorer_count; scorer_count+=1
            original_reward=original_multi_score(*args,**kwargs)
            scorer_lock=threading.Lock()
            def recorded_reward(images,prompts,*args,**kwargs):
                # The trainer's outer pool otherwise re-enters one Paddle scorer.
                with scorer_lock:
                    image_hash=runtime.tensor_digest([images])
                    details,extra=original_reward(images,prompts,*args,**kwargs)
                    runtime.result.setdefault('reward_fingerprints',[]).append({
                        'scorer':scorer_index,'images':image_hash,
                        'prompts':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
                        'rewards':hashlib.sha256(json.dumps(details,sort_keys=True,allow_nan=False).encode()).hexdigest()})
                return details,extra
            return recorded_reward
        flow_grpo.rewards.multi_score=multi_score

    def configure(self,config):
        p=self.profile
        model_root=Path(os.environ['FRONTIER_MODEL_ROOT'])
        config.pretrained.model=str(model_root/'stable-diffusion-3.5-medium')
        config.train.kl_ref_lora_path=str(model_root/'SD3.5M-FlowGRPO-Text')
        config.dataset=str(Path(p['dataset']).resolve())
        config.sample.train_batch_size=p['train_batch_size']
        config.sample.num_image_per_prompt=p['repeat_count']
        config.sample.num_batches_per_epoch=p['batches_per_epoch']
        config.sample.test_batch_size=p['evaluation_batch_size']
        config.train.batch_size=p['train_microbatch_size']
        config.train.gradient_accumulation_steps=p['gradient_accumulation_steps']
        config.seed=p['training_seed']; config.mixed_precision=p['mixed_precision']
        config.logdir=str(self.output/'logs'); config.save_dir=str(self.output)
        config.run_name='bounded_ocr'
        required=[config.resolution==512,config.sample.num_steps==10,
                  config.sample.eval_num_steps==40,config.sample.guidance_scale==4.5,
                  config.train.reward_mode=='kl_only',config.train.beta==0.04,
                  config.train.timestep_fraction==0.99,config.train.ema,
                  not config.train.mar_lora,not config.train.lora_path,
                  dict(config.reward_fn)=={'ocr':1.0}]
        if not all(required): raise ValueError('Frozen scientific configuration changed.')
        (self.output/'effective_config.json').write_text(config.to_json())

    def observe_optimizer(self,optimizer,parameters):
        torch=self.torch; parameters=list(parameters)
        self.initial=[p.detach().cpu().clone() for p in parameters]
        self.result['initial_parameter_sha256']=self.tensor_digest(parameters)
        self.result['optimizer_parameter_sha256']=[]
        self.result['gradient_sha256']=[]
        previous=[]
        def before(optimizer,args,kwargs):
            if self.updates>=self.profile['updates']:
                raise ValueError('Update budget crossed before a complete epoch/EMA boundary.')
            grads=[p.grad for p in parameters if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise ValueError('Missing or nonfinite gradients.')
            self.result['gradient_sha256'].append(self.tensor_digest(grads))
            previous[:]=[p.detach().clone() for p in parameters]
        def after(optimizer,args,kwargs):
            if not any(not torch.equal(p.detach(),old) for p,old in zip(parameters,previous)):
                raise ValueError('Optimizer did not change trainable weights.')
            if not all(torch.isfinite(p).all() for p in parameters):
                raise ValueError('Nonfinite updated parameters.')
            previous.clear(); self.updates+=1
            self.result['optimizer_parameter_sha256'].append(self.tensor_digest(parameters))
            self.result['verified_optimizer_updates']=self.updates
            self.write_progress()
        optimizer.register_step_pre_hook(before); optimizer.register_step_post_hook(after)

    def bind(self,local,namespace):
        torch=self.torch; self.namespace=namespace; self.local=local
        from frontier_qualification import state_digest, TransitionProbe, deterministic_slots
        model=local['accelerator'].unwrap_model(local['transformer'])
        self.initial_base_digest=state_digest({name:value for name,value in model.named_parameters()
                                               if '.lora_' not in name})
        self.result['bound_cpu_rng_sha256']=self.tensor_digest([torch.get_rng_state()])
        self.result['bound_cuda_rng_sha256']=self.tensor_digest([torch.cuda.get_rng_state()])
        self.result['rollout_fingerprints']=[]
        self.result['evaluation_image_sha256']=[]
        pipeline=local['pipeline']; transformer=local['transformer']; teacher=local['ref_transformer']
        # Teacher is inference-only in the released path; set_adapter may have set
        # LoRA requires_grad back on. Freezing has no forward numerical effect.
        teacher.requires_grad_(False)
        def teacher_hook(module,args,kwargs): self.teacher_forwards+=1
        def student_hook(module,args,kwargs):
            self.student_forwards+=1
            if not self.evaluation and getattr(module,'disable_adapters',False): self.base_forwards+=1
        teacher.register_forward_pre_hook(teacher_hook,with_kwargs=True)
        transformer.register_forward_pre_hook(student_hook,with_kwargs=True)
        original=namespace['pipeline_with_logprob']
        original_step=namespace['_sde_step']
        step_signature=inspect.signature(original_step)
        probe=TransitionProbe(self,original_step)
        guard=load_transition_guard(self.profile)
        checked={'sampled':0, 'replayed':0}
        if guard is not None:
            self.result['transition_validation']={'tolerance':self.profile['transition_tolerance'],
                                                  'checked_calls':checked}
        def normalized_step(*args,**kwargs):
            bound=step_signature.bind(*args,**kwargs)
            bound.apply_defaults()
            return deterministic_slots(original_step(*args,**kwargs), bound.arguments['noise_level'])
        def validate_step(result,args,kwargs,kind):
            if guard is not None:
                bound=step_signature.bind(*args,**kwargs)
                bound.apply_defaults()
                values=bound.arguments
                noise=values['noise_level']
                guard(normalized_step,values['self'],values['model_output'],
                      values['timestep'],values['sample'],observed=result,noise_level=noise,
                      tolerance=self.profile['transition_tolerance'],require_gradient=bool(noise))
                checked[kind]+=1
        def sampled_step(*args,**kwargs):
            result=probe(*args,**kwargs)
            validate_step(result,args,kwargs,'sampled')
            return result
        pipeline_module=importlib.import_module('flow_grpo.diffusers_patch.sd3_pipeline_with_logprob')
        pipeline_module.sde_step_with_logprob=sampled_step
        self.transition_probe=probe
        def audited_step(*args,**kwargs):
            result=original_step(*args,**kwargs)
            if len(result) not in (4,5): raise ValueError('Invalid step tuple.')
            next_state,logp,mean,coefficient=result[:4]
            if not all(torch.isfinite(x).all() for x in (next_state,logp,mean,coefficient)):
                raise ValueError('Nonfinite consumed transition.')
            if not (coefficient>0).all(): raise ValueError('Invalid consumed KL coefficient.')
            if kwargs.get('prev_sample') is not None and not torch.equal(next_state,kwargs['prev_sample'].float()):
                raise ValueError('Replay changed the supplied next state.')
            validate_step(result,args,kwargs,'replayed')
            return result
        namespace['_sde_step']=audited_step
        namespace['sde_step_with_logprob']=audited_step
        def audited_pipeline(pipe,*args,**kwargs):
            if self.evaluation:
                batch=len(kwargs['prompt_embeds'])
                rows=self.profile['evaluation'][self.eval_index:self.eval_index+batch]
                if len(rows)!=batch: raise ValueError('Evaluation batch exceeds frozen suite.')
                generators=[torch.Generator(device=pipe._execution_device).manual_seed(r['seed']) for r in rows]
                # Use the pipeline's authentic initial-latent preparation and reset
                # it independently of training RNG consumption for every workload.
                latents=pipe.prepare_latents(batch,pipe.transformer.config.in_channels,512,512,
                    kwargs['prompt_embeds'].dtype,pipe._execution_device,generators,None).float()
                kwargs['latents']=latents
                kwargs['generator']=generators
                self.eval_latent_hashes.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in latents)
                self.eval_index+=batch
            else:
                self.rollouts+=1
            result=original(pipe,*args,**kwargs)
            images,latents,logps=result[:3]
            if self.evaluation:
                self.result['evaluation_image_sha256'].extend(self.tensor_digest([x]) for x in images)
            else:
                self.result['rollout_fingerprints'].append({
                    'prompt_embeddings':self.tensor_digest([kwargs['prompt_embeds'],kwargs['pooled_prompt_embeds']]),
                    'initial_latents':self.tensor_digest([latents[0]]),
                    'final_latents':self.tensor_digest([latents[-1]]),
                    'images':self.tensor_digest([images])})
            steps=kwargs['num_inference_steps']
            if len(latents)!=steps+1 or len(logps)!=steps:
                raise ValueError('Incomplete ordered trajectory.')
            if images.shape[-2:]!=(512,512) or not torch.isfinite(images).all():
                raise ValueError('Invalid decoded image.')
            if not all(torch.isfinite(x).all() for x in latents):
                raise ValueError('Nonfinite trajectory.')
            if not self.evaluation:
                if not all(x.shape==(images.shape[0],) and torch.isfinite(x).all() for x in logps):
                    raise ValueError('Invalid rollout likelihoods.')
            if self.evaluation and not all(bool((x==0).all()) for x in logps):
                raise ValueError('Deterministic sampled log slots must be zero on support.')
            return result
        namespace['pipeline_with_logprob']=audited_pipeline

    def tensor_digest(self,values):
        digest=hashlib.sha256()
        for value in values:
            value=value.detach().cpu().contiguous()
            digest.update(str((str(value.dtype),tuple(value.shape))).encode())
            digest.update(value.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def write_progress(self):
        self.result['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        tmp=self.output/'progress.tmp'
        tmp.write_text(json.dumps(self.result,allow_nan=False))
        tmp.replace(self.output/'progress.json')

    def boundary(self,local,namespace):
        if time.monotonic()-self.start>self.profile['max_runner_seconds']:
            raise TimeoutError('Bounded run expired.')
        if self.updates==0:
            raise ValueError('No optimizer update survived retained OCR filtering.')
        if self.updates<self.profile['updates']: return
        if self.updates!=self.profile['updates'] or not local['accelerator'].sync_gradients:
            raise ValueError('Invalid endpoint update boundary.')
        torch=self.torch
        if not self.rollouts or not self.teacher_forwards or not self.loss_calls:
            raise ValueError('Missing authentic computation.')
        params=local['transformer_trainable_parameters']
        self.result['endpoint_parameter_sha256']=self.tensor_digest(params)
        self.result['endpoint_ema_sha256']=self.tensor_digest(local['ema'].ema_parameters)
        if not any(not torch.equal(p.detach().cpu(),old) for p,old in zip(params,self.initial)):
            raise ValueError('Trainable state unchanged.')
        # Save complete recoverable state before endpoint evaluation. Preserve on
        # failure; resumption policy is controlled by the later qualification run.
        accelerator=local['accelerator']
        # Accelerate 1.4 otherwise ignores output_dir for this upstream config.
        naming=accelerator.project_configuration.automatic_checkpoint_naming
        try:
            accelerator.project_configuration.automatic_checkpoint_naming=False
            accelerator.save_state(str(self.output/'state'))
        finally:
            accelerator.project_configuration.automatic_checkpoint_naming=naming
        torch.save({'ema':local['ema'].state_dict(),'epoch':local['epoch'],
                    'global_step':local['global_step'],'verified_updates':self.updates,
                    'ema_update_step_interval':local['ema'].update_step_interval},self.output/'state/endpoint.pt')
        namespace['save_ckpt'](str(self.output),local['transformer'],local['global_step'],accelerator,
                              local['ema'],params,local['config'])
        from frontier_qualification import verify_checkpoint, reload_ema_adapter
        verify_checkpoint(self,local)
        self.evaluation=True
        original_reward=local['eval_reward_fn']
        def capture(images,prompts,metadata,**kwargs):
            details,extra=original_reward(images,prompts,metadata,**kwargs)
            start=len(self.rewards); expected=self.profile['evaluation'][start:start+len(prompts)]
            if [p['prompt'] for p in expected]!=prompts:
                raise ValueError('Evaluation loader order or membership changed.')
            self.rewards.extend(float(x) for x in details['ocr'])
            return details,extra
        namespace['eval'](local['pipeline'],local['test_dataloader'],local['text_encoders'],local['tokenizers'],
                          local['config'],accelerator,local['global_step'],capture,local['executor'],
                          local['autocast'],local['num_train_timesteps'],local['ema'],params)
        if len(self.rewards)!=len(self.profile['evaluation']):
            raise ValueError('Evaluation coverage incomplete.')
        first_rewards=self.rewards[:]
        first_latents=self.eval_latent_hashes[:]
        first_images=self.result['evaluation_image_sha256'][:]
        reloaded=reload_ema_adapter(self,local)
        self.rewards=[]; self.eval_latent_hashes=[]; self.eval_index=0
        self.result['evaluation_image_sha256']=[]
        namespace['eval'](local['pipeline'],local['test_dataloader'],local['text_encoders'],local['tokenizers'],
                          local['config'],accelerator,local['global_step'],capture,local['executor'],
                          local['autocast'],local['num_train_timesteps'],local['ema'],params)
        reload_receipt=self.result['ema_adapter_reload']
        reload_receipt.update(decoded_images_equal=first_images==self.result['evaluation_image_sha256'],
                              ocr_equal=first_rewards==self.rewards,
                              initial_latents_equal=first_latents==self.eval_latent_hashes)
        if not all(reload_receipt[key] for key in ('decoded_images_equal','ocr_equal','initial_latents_equal')):
            raise ValueError('Reloaded EMA endpoint evaluation differs.')
        reloaded.set_adapter('default'); reloaded.delete_adapter('frontier_reload')
        self.rewards=first_rewards; self.eval_latent_hashes=first_latents
        self.result['evaluation_image_sha256']=first_images
        self.result['zero_noise_slots_normalized']=self.transition_probe.normalized_slots
        self.result.update(complete=True,ocr_rewards=self.rewards,
                           initial_evaluation_latent_sha256=self.eval_latent_hashes,
                           rollout_batches=self.rollouts,teacher_forwards=self.teacher_forwards,
                           student_forwards=self.student_forwards,backward_calls=self.loss_calls,
                           elapsed_seconds=time.monotonic()-self.start,
                           max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
        # These diagnostics are NOT a trusted validity attestation. Independent
        # hidden guards and checkpoint reload qualification are still required.
        self.result['independent_validity_qualified']=False
        self.write_progress(); raise Completed()


def upstream_launch(args):
    if args!=LAUNCH_ARGS: raise ValueError('Unexpected original launcher arguments.')
    profile=read_profile(os.environ['FRONTIER_RUN_PROFILE'])
    # Runtime installs reward hooks; bind imports to the patched tree first.
    sys.path.insert(0,str(Path.cwd()))
    runtime=Runtime(profile,os.environ['FRONTIER_RUN_OUTPUT'])
    source=Path(TRAINER).read_text(); tree,adaptations=transform_trainer(source)
    (runtime.output/'runtime_adaptations.json').write_text(json.dumps(adaptations,indent=2))
    namespace={'__name__':'__main__','__file__':str(Path(TRAINER).resolve()),'_frontier':runtime}
    sys.argv=[TRAINER,'--config','config/grpo.py:general_ocr_sd3_8gpu_opd']
    try: exec(compile(tree,TRAINER,'exec'),namespace)
    except Completed: return
    finally: runtime.write_progress()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--upstream-launch':
        upstream_launch(sys.argv[2:]); return
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    p=read_profile(args.profile)
    adapter=Path(__file__).resolve().parent
    env=os.environ.copy()
    env.update(FRONTIER_RUNTIME_ADAPTER=str(adapter),FRONTIER_RUN_PROFILE=str(Path(args.profile).resolve()),
               FRONTIER_RUN_OUTPUT=str(Path(args.output).resolve()),HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONDONTWRITEBYTECODE='1')
    env['PATH']=str(adapter/'frontier_shim')+os.pathsep+env['PATH']
    subprocess.run(['bash',ENTRYPOINT],check=True,env=env,timeout=p['max_runner_seconds'])

if __name__=='__main__':
    try: main()
    except Exception:
        # The judge captures stderr privately; train feedback never includes it.
        import traceback
        traceback.print_exc()
        raise SystemExit(2)
