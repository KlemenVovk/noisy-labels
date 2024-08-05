import argparse
from importlib.machinery import SourceFileLoader

def run(method_config):
    model, datamodule, trainer = method_config.build_modules()
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config .py file.', default="configs/methods/cores2/cores_cifar10_clean.py", required=False)
    parser.add_argument('--method', type=str, help='Name of MethodConfig class to run.', default="cores_cifar10_clean", required=False)
    args = parser.parse_args()

    modulename = SourceFileLoader("cfg", str(args.config)).load_module()
    method_config = getattr(modulename, args.method)
    run(method_config)
