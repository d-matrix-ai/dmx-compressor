def _get_submodule(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target_name, target


def _set_submodule(parent, target_name, target):
    setattr(parent, target_name, target)


def transform_submodule(model, key, fn):
    parent, target_name, target = _get_submodule(model, key)
    transformed_target = fn(target)
    _set_submodule(parent, target_name, transformed_target)
