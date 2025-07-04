import importlib

def create_loss(loss_name: str, **kwargs):
    loss_cls = _import_class(f'edepth_rewrite.losses.{loss_name}')
    return loss_cls(**kwargs)

def _import_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name) 