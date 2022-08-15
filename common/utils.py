from ast import arg
import numpy as np
import inspect
import functools


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args): #这一块是怎样改的
    '''获取场景以及对应的维度，但是这里的所以可以进行初始化
    '''

    from ENV import env_
    env = env_.Cenv(args)#初始化环境
    args.n_agents = args.APs_length #训练三个AP


    # args.APofEndDevices 里面存储每个AP相连的EndDevice个数
    # 将state设置为 功率 + [带宽段数量] （为0表示不需要该段频谱 为1表示需要该段频谱） 准备先让每个用户只打算接入一个频段 其实就是用独热编码
    args.obs_shape = [(1+args.numofBand)*args.APofEndDevices[i] for i in range(args.n_agents)]  # 每一维代表该agent的obs维度 观测值的维度
    ###将用户需求如何转换成state 这个用户需求怎样写
    ''''''#action同state一样 
    args.action_shape = [(1+args.numofBand)*args.APofEndDevices[i] for i in range(args.n_agents)]  #其实action是一样的 
    ###action 又应该是什么
    args.high_action = 1
    args.low_action = -1
    return env, args
