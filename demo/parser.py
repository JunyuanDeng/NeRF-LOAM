import yaml
import argparse

class ArgumentParserX(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_argument("config", type=str)

    def parse_args(self, args=None, namespace=None):
        _args = self.parse_known_args(args, namespace)[0]
        file_args = argparse.Namespace()
        file_args = self.parse_config_yaml(_args.config, file_args)
        file_args = self.convert_to_namespace(file_args)
        for ckey, cvalue in file_args.__dict__.items():
            try:
                self.add_argument('--' + ckey, type=type(cvalue),
                                  default=cvalue, required=False)
            except argparse.ArgumentError:
                continue
        _args = super().parse_args(args, namespace)
        return _args

    def parse_config_yaml(self, yaml_path, args=None):

        with open(yaml_path, 'r') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        if configs is not None:
            base_config = configs.get('base_config')
            if base_config is not None:
                base_config = self.parse_config_yaml(configs["base_config"])
                if base_config is not None:
                    configs = self.update_recursive(base_config, configs)
                else:
                    raise FileNotFoundError("base_config specified but not found!")

        return configs

    def convert_to_namespace(self, dict_in, args=None):
        if args is None:
            args = argparse.Namespace()
        for ckey, cvalue in dict_in.items():
            if ckey not in args.__dict__.keys():
                args.__dict__[ckey] = cvalue

        return args

    def update_recursive(self, dict1, dict2):
        for k, v in dict2.items():
            if k not in dict1:
                dict1[k] = dict()
            if isinstance(v, dict):
                self.update_recursive(dict1[k], v)
            else:
                dict1[k] = v
        return dict1

def get_parser():
    parser = ArgumentParserX()
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument("--debug", action='store_true')
    return parser

if __name__ == '__main__':
    args = ArgumentParserX()
    print(args.parse_args())
