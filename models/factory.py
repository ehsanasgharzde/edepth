import importlib

def create_model(backbone: str, decoder: str, head: str = None, **kwargs):
    backbone_cls = _import_class(f'edepth_rewrite.models.backbones.{backbone}')
    decoder_cls = _import_class(f'edepth_rewrite.models.decoders.{decoder}')
    head_cls = _import_class(f'edepth_rewrite.models.heads.{head}') if head else None
    backbone = backbone_cls(**kwargs.get('backbone', {}))
    decoder = decoder_cls(**kwargs.get('decoder', {}))
    head = head_cls(**kwargs.get('head', {})) if head_cls else None
    return backbone, decoder, head

def _import_class(path):
    module_name, class_name = path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name) 