from header import *

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        if args['stage'] == 2:
            self.load_stage_1_parameters(args["delta_ckpt_path"])
            print(f'[!] load stage 1 checkpoint from {args["delta_ckpt_path"]}')

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        # ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        # ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps'] / ds_params['train_micro_batch_size_per_gpu']
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps'] / 2
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']) / 8)
        if self.args['world_size'] * ds_params['gradient_accumulation_steps'] * ds_params['train_micro_batch_size_per_gpu'] != ds_params['train_batch_size']:
            print("Force setting train batch size")
            ds_params['train_batch_size'] = self.args['world_size'] * ds_params['gradient_accumulation_steps'] * ds_params['train_micro_batch_size_per_gpu']
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)
        # print("Begin backward, {}".format(int(os.getenv('RANK', '0'))))
        self.ds_engine.backward(loss)
        # print("After backward, {}".format(int(os.getenv('RANK', '0'))))
        self.ds_engine.step()
        # print("After optimizer step, {}".format(int(os.getenv('RANK', '0'))))
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc

    @torch.no_grad()
    def valid_model(self, batch):
        self.model.eval()
        loss, mle_acc = self.ds_engine(batch)
        return loss.item(), mle_acc

    def _zero3_consolidated_16bit_state_dict(self):
        """
        Get a full non-partitioned state_dict with fp16 weights on cpu.
        Important: this function must be called on all ranks and not just rank 0.
        This is similar to nn.Module.state_dict (modelled after _save_to_state_dict), but:
        1. consolidates the weights from different partitions on gpu0
        2. works on one layer at a time to require as little gpu0 memory as possible, by
        moving the already consolidated weights to cpu
        3. takes care to keep the shared params shared when gradually copying the params to cpu
        Returns:
            a consolidated fp16 ``state_dict`` on cpu on rank 0, ``None`` on other ranks
        """
        state_dict = OrderedDict()
        shared_params = {}

        def get_layer_state_dict(module, prefix=""):
            # gather one layer at a time to be memory-efficient
            # must use modifier_rank=0 to release GPU memory after each layer gathered
            #see_memory_usage("before GatheredParameters", force=True)
            with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
                if int(os.getenv('RANK', '0')) == 0:
                    # handle params
                    for name, param in module.named_parameters(recurse=False):
                        if param is None:
                            continue
                        key = prefix + name
                        if param.requires_grad:
                            # can't rely on param.data_ptr() as it will be reused as weights gets
                            # gathered and reduced, but param.ds_id is unique across all zero weights
                            # (and shared params will have the same param.ds_id)
                            if param.ds_id in shared_params:
                                # shared weights
                                # print(f"`{key}` is shared with `{shared_params[param.ds_id]}`")
                                state_dict[key] = state_dict[shared_params[param.ds_id]]
                            else:
                                state_dict[key] = param.detach().cpu()
                                shared_params[param.ds_id] = key
                        # print(f"param {param.ds_id} {param.shape} {key} ")

                    # now buffers - not sure if need to take care of potentially shared weights here
                    for name, buf in module.named_buffers(recurse=False):
                        if (buf is not None and name not in module._non_persistent_buffers_set):
                            state_dict[prefix + name] = buf.detach().cpu()
                    #see_memory_usage("after GatheredParameters", force=True)

            for name, child in module.named_children():
                if child is not None:
                    get_layer_state_dict(child, prefix + name + ".")

        # Prepare for checkpoint save by ensuring all parameters are partitioned
        self.optimizer.checkpoint_event_prologue()

        # see_memory_usage("before get_layer_state_dict", force=False)
        get_layer_state_dict(self.ds_engine.module, prefix="")
        # see_memory_usage("after get_layer_state_dict", force=False)

        # self.ds_engine.optimizer.checkpoint_event_epilogue()

        return state_dict
    
    def save_model(self, path, current_step):
        # only save trainable model parameters
        if self.ds_engine.zero_gather_16bit_weights_on_model_save():
            # state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
            checkpoint = self._zero3_consolidated_16bit_state_dict()
        else:
            checkpoint = OrderedDict()
            for k, v in self.ds_engine.module.named_parameters():
                if v.requires_grad:
                    checkpoint[k] = v
        if int(os.getenv('RANK', '0')) == 0:
            torch.save(checkpoint, f'{path}/pytorch_model_{current_step}.pt')
            # save tokenizer
            self.model.llama_tokenizer.save_pretrained(path)
            # save configuration
            self.model.llama_model.config.save_pretrained(path)
            print(f'[!] save model into {path}')

    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
