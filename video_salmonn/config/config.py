from omegaconf import OmegaConf

class Config:
    def __init__(self, args):
        self.config = {}

        self.args = args
        user_config = self._build_opt_list(self.args.options)
        config = OmegaConf.load(self.args.cfg_path)
        config = OmegaConf.merge(config, user_config)
        self.config = config

    def _convert_to_dot_list(self, opts):
        if opts is None:
            opts = []

        if len(opts) == 0:
            return opts

        has_equal = opts[0].find("=") != -1

        if has_equal:
            return opts

        return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

    def _build_opt_list(self, opts):
        opts_dot_list = self._convert_to_dot_list(opts)
        return OmegaConf.from_dotlist(opts_dot_list)
