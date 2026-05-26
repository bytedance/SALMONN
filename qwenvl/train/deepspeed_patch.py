
import deepspeed
import torch
if deepspeed.__version__ == "0.17.5" or deepspeed.__version__ == "0.18.0":
    from deepspeed.runtime.zero.parameter_offload import *
    def _register_deepspeed_module(self, module, count=[0]):
        my_count = count[0]
        module.ds_id = my_count

        #print(f"{module.__class__} : {module.ds_id}")

        if z3_leaf_module(module):
            for param in module.parameters():
                param.ds_z3_leaf_module = module
        else:
            for child in module.children():
                count[0] = count[0] + 1
                self._register_deepspeed_module(child, count=count)

        @torch.compiler.disable
        def _pre_forward_module_hook(module, *args):
            self.pre_sub_module_forward_function(module)

        @instrument_w_nvtx
        def _post_forward_module_hook(module, input, output):

            global FWD_MODULE_STACK
            FWD_MODULE_STACK.pop()
            if output is None:
                output = []
            elif not isinstance(output, (list, tuple)):
                if torch.is_tensor(output):
                    output = [output]
                else:
                    #print(f'got UNKNOWN type {type(output)}')
                    outputs = []
                    output = output if isinstance(output, dict) else vars(output)
                    for name, val in output.items():
                        if not name.startswith('__') and torch.is_tensor(val):
                            outputs.append(val)
                    output = outputs

            for item in filter(lambda item: is_zero_param(item) or hasattr(item, 'ds_param_alias'), output):
                key = id(item) if hasattr(item, 'ds_id') else id(item.ds_param_alias)
                actual_external_param = item if hasattr(item, 'ds_id') else item.ds_param_alias

                if not any(key in m._external_params for m in FWD_MODULE_STACK):
                    actual_external_param.is_external_param = True
                    module_to_register = FWD_MODULE_STACK[-1]
                    register_external_parameter(module_to_register, actual_external_param)
                    print_rank_0(
                        f'Registering dangling parameter for module {module_to_register.__class__.__name__}, ds_id = {actual_external_param.ds_id}.',
                        force=False)

                    # It's possible that the parameter was already external to the completed module. If so, remove it the
                    # registration as it will be covered by the outer module instead.
                    if key in module._external_params:
                        print_rank_0(
                            f'  Unregistering nested dangling parameter from module {module.__class__.__name__}, ds_id = {actual_external_param.ds_id}',
                            force=False)
                        unregister_external_parameter(module, actual_external_param)

                    actual_external_param.all_gather()

            self.post_sub_module_forward_function(module)

        def _bwd_hook_unexpected_inputs_msg(value):
            return f"A module has unknown inputs or outputs type ({type(value)}) and the tensors embedded in it cannot be detected. " \
                "The ZeRO-3 hooks designed to trigger before or after backward pass of the module relies on knowing the input and " \
                "output tensors and therefore may not get triggered properly."

        def _pre_backward_module_hook(module, inputs, output):

            return apply_to_tensors_only(module.pre_bwd_fn.apply,
                                        output,
                                        warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        #This is an alternate to doing _post_backward_module_hook
        #it uses tensor.register_hook instead of using torch.autograd.Function
        def _alternate_post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            #print(f"Before Forward {module.__class__.__name__}")

            def _run_after_backward_hook(*unused):
                module.ds_grads_remaining = module.ds_grads_remaining - 1
                if module.ds_grads_remaining == 0:
                    #print(f"After backward {module.__class__.__name__}")
                    self.post_sub_module_backward_function(module)

            def _run_before_forward_function(input):
                if input.requires_grad:
                    module.ds_grads_remaining += 1

            return _apply_forward_and_backward_to_tensors_only(module, _run_before_forward_function,
                                                            _run_after_backward_hook, inputs)

        @torch.compiler.disable
        def _post_backward_module_hook(module, inputs):
            module.ds_grads_remaining = 0

            return apply_to_tensors_only(module.post_bwd_fn.apply,
                                        inputs,
                                        warning_msg_fn=_bwd_hook_unexpected_inputs_msg)

        # Pre forward hook
        self.forward_hooks.append(module.register_forward_pre_hook(_pre_forward_module_hook))

        # Post forward hook
        self.forward_hooks.append(module.register_forward_hook(_post_forward_module_hook))

        # Pre backward hook
        if not hasattr(module, "pre_bwd_fn"):

            @instrument_w_nvtx
            def _run_before_backward_function(sub_module):
                # some models (e.g. Albert) may run multiple forwards on the same layer in a loop
                # before doing backwards, so each backward will need a pre-fetch - using reference
                # counting to support this scenario
                #print(f"COUNTER before: {sub_module.applied_pre_backward_ref_cnt}")
                if sub_module.applied_pre_backward_ref_cnt > 0:
                    self.pre_sub_module_backward_function(sub_module)
                    sub_module.applied_pre_backward_ref_cnt -= 1
                #print(f"COUNTER after: {sub_module.applied_pre_backward_ref_cnt}")

            class PreBackwardFunctionForModule(torch.autograd.Function):

                @staticmethod
                def forward(ctx, outputs):
                    # Capture `module` and _run_before_backward_function
                    ctx.module = module
                    ctx.pre_backward_function = _run_before_backward_function
                    if not hasattr(ctx.module, "applied_pre_backward_ref_cnt"):
                        ctx.module.applied_pre_backward_ref_cnt = 0
                    ctx.module.applied_pre_backward_ref_cnt += 1
                    outputs = outputs.detach()
                    return outputs

                @staticmethod
                def backward(ctx, *args):
                    ctx.pre_backward_function(ctx.module)
                    return args

            module.pre_bwd_fn = PreBackwardFunctionForModule

        self.backward_hooks.append(module.register_forward_hook(_pre_backward_module_hook))

        # post backward hook
        if "post_bwd_fn" not in module.__dict__:

            @instrument_w_nvtx
            def _run_after_backward_function(sub_module):
                if sub_module.ds_grads_remaining == 0:
                    self.post_sub_module_backward_function(sub_module)

            class PostBackwardFunctionModule(torch.autograd.Function):

                @staticmethod
                def forward(ctx, output):
                    ctx.module = module
                    if output.requires_grad:
                        #TODO SOME TIMES post backward does not seem to be triggered debug in detail
                        #Should only cause increase in memory not correctness issue
                        #if output.grad_fn.__class__.__name__ == 'ViewBackward':
                        #    ctx.view=True
                        #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
                        #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
                        #if module.ds_grads_remaining == 0:
                        #    print(f"Before Forward: {ctx.module.__class__.__name__}")
                        module.ds_grads_remaining += 1
                        ctx.post_backward_function = _run_after_backward_function
                    output = output.detach()
                    return output

                @staticmethod
                def backward(ctx, *args):
                    ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
                    if ctx.module.ds_grads_remaining == 0:
                        ctx.post_backward_function(ctx.module)
                    return args

            module.post_bwd_fn = PostBackwardFunctionModule

        self.backward_hooks.append(module.register_forward_pre_hook(_post_backward_module_hook))

    deepspeed.runtime.zero.DeepSpeedZeRoOffload._register_deepspeed_module = _register_deepspeed_module