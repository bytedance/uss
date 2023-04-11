from functools import partial


def linear_warm_up(step, warm_up_steps: int, reduce_lr_steps: int):
    r"""Get lr_lambda for LambdaLR. E.g.,

    .. code-block: python
        lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

        from torch.optim.lr_scheduler import LambdaLR
        LambdaLR(optimizer, lr_lambda)

    Args:
        warm_up_steps: int, steps for warm up
        reduce_lr_steps: int, reduce learning rate by 0.9 every #reduce_lr_steps steps

    Returns:
        learning rate: float
    """
    if step <= warm_up_steps:
        lr_scale = step / warm_up_steps
    else:
        lr_scale = 0.9 ** (step // reduce_lr_steps)

    return lr_scale



def constant_warm_up(step, warm_up_steps, reduce_lr_steps):
    
    if 0 <= step < warm_up_steps:
        lr_scale = 0.001

    elif warm_up_steps <= step < 2 * warm_up_steps:
        lr_scale = 0.01

    elif 2 * warm_up_steps <= step < 3 * warm_up_steps:
        lr_scale = 0.1

    else:
        lr_scale = 1

    return lr_scale


def get_lr_lambda(lr_lambda_type, **kwargs):

    if lr_lambda_type == "constant_warm_up":

        lr_lambda_func = partial(
            constant_warm_up, 
            warm_up_steps=kwargs["warm_up_steps"], 
            reduce_lr_steps=kwargs["reduce_lr_steps"],
        )

    elif lr_lambda_type == "linear_warm_up":

        lr_lambda_func = partial(
            linear_warm_up, 
            warm_up_steps=kwargs["warm_up_steps"], 
            reduce_lr_steps=kwargs["reduce_lr_steps"],
        )

    else:
        raise NotImplementedError

    return lr_lambda_func
