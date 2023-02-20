from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os
from runner.basictrainner import Trainer
from lib.metric import TweedieLoss
from lib.traininits import init_seed, print_model_parameters
from lib.config_handler import YamlHandler
from lib.logger import get_logger

############################################################################
Mode = 'train'
DATASET = 'battery_cost'
model_name = 'Informer'  # Transformer, GRU_CNN, encoder_Transformer, Informer, TCN, TFT, logspare_transformer
############################################################################

config_file = './runner/{}.yaml'.format(DATASET)
config = YamlHandler(config_file).read_yaml()

init_seed(config.base.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    device = 'cpu'

if model_name == 'GRU_CNN':
    from model.gru_cnn import GRU as network
    model = network(**config.model.GRU_CNN)
elif model_name == 'gru_att':
    from model.gru_att import GruModel as network
    model = network()
elif model_name == 'Transformer':
    from model.Transformer import TransformerTimeSeries as network
    model = network(**config.model.Transformer)
elif model_name == 'encoder_Transformer':
    from model.encoder_Transformer import TransformerTimeSeries as network
    model = network(**config.model.encoder_Transformer)
elif model_name == 'Informer':
    from model.Informer import Informer as network
    model = network(**config.model.Informer)
elif model_name == 'TCN':
    from model.TCN import TCN as network
    model = network(**config.model.TCN)
elif model_name == 'TFT':
    from model.TFT import TFT as network
    model = network(**config.model.TFT)
elif model_name == 'logspare_transformer':
    from model.logspare_transformer import DecoderTransformer as network
    model = network(**config.model.logspare_transformer)

# set the number of gup_devices to train
model = nn.DataParallel(model, device_ids=[0]).cuda()  # 0, 1, 2, 3, 4, 5, 6, 7

if config.train.optim == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=config.train.lr_init, weight_decay=0)
elif config.train.optim == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=config.train.lr_init, weight_decay=0)
#learning rate decay
lr_scheduler = None
if config.train.lr_decay:
    T_max = int(config.train.get('epochs'))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)
#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
task_names = ['order_cnt_per_bike', 'battery_cost_per_order', 'distance_per_bike', 'battery_cost_per_bike']
log_dir = os.path.join(current_dir,'checkpoints', task_names[config.data.label_index], model_name, current_time)
config.log.log_dir = log_dir
# log
if os.path.isdir(config.log.log_dir) == False and not config.base.debug:
    os.makedirs(config.log.log_dir, exist_ok=True)
logger = get_logger(config.log.log_dir, name=model_name, debug=config.base.debug)

print_model_parameters(model, logger, only_num=False)

print(torch.__version__)

#init loss function, optimizer
if config.train.loss == 'tweedie':
    loss = TweedieLoss(p=1.4)
elif config.train.loss == 'mae':
    loss = torch.nn.L1Loss()
elif config.train.loss == 'mse':
    loss = torch.nn.MSELoss()
elif config.train.loss == "poisson":
    loss = nn.PoissonNLLLoss(log_input=True, reduction='none', full=False)
else:
    raise ValueError
#start training
current = bool(config.model.get('{}'.format(model_name)).get('current'))
day = bool(config.model.get('{}'.format(model_name)).get('day'))
# current, day = False, False  # for gru_att baseline
trainer = Trainer(config, model, model_name, loss, optimizer, DATASET, current, day, logger, lr_scheduler=lr_scheduler)
if Mode == 'train':
    trainer.train()
elif Mode == 'test':
    model.load_state_dict(torch.load('./experiments/battery_cost_per_bike/logspare_transformer/20220623201924/best_model.pth'))
    logger.info("Load saved model")
    trainer.test(config, model,model_name, config.data.test_dir, logger, current=current, day=day, log_dir=log_dir)
else:
    raise ValueError