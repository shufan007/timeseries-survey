import argparse, configparser

def load_args(config_file, model):
    config = configparser.ConfigParser()
    config.read(config_file)
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default=model, type=str)
    #data
    parser.add_argument('--train_len', default=config['data']['train_len'],type=int)
    parser.add_argument('--test_len', default=config['data']['test_len'],type=int)
    parser.add_argument('--train_dir', default=config['data']['train_dir'], type=str)
    parser.add_argument('--test_dir', default=config['data']['test_dir'], type=str)
    parser.add_argument('--label_index', default=config['data']['label_index'],type=int)
    #model
    parser.add_argument('--cate_shape', default=config['model']['cate_shape'], type=int)
    parser.add_argument('--cate_size', default=config['model']['cate_size'], type=int)
    parser.add_argument('--seq_dim', default=config['model']['seq_dim'], type=int)
    parser.add_argument('--flat_dim', default=config['model']['flat_dim'], type=int)
    parser.add_argument('--cate_hidden', default=config['model']['cate_hidden'], type=int)
    parser.add_argument('--hidden_dim', default=config['model']['hidden_dim'], type=int)
    parser.add_argument('--extra_flat_dim', default=config['model']['extra_flat_dim'], type=int)
    parser.add_argument('--static_size', default=config['model']['static_size'], type=int)
    parser.add_argument('--mha_hidden', default=config['model']['mha_hidden'], type=int)
    parser.add_argument('--n_head', default=config['model']['n_head'], type=int)

    #train
    parser.add_argument('--debug', default=config['train']['debug'], type=eval)
    parser.add_argument('--optim', default=config['train']['optim'], type=str)
    parser.add_argument('--loss', default=config['train']['loss'], type=str)
    parser.add_argument('--seed', default=config['train']['seed'], type=int)
    parser.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser.add_argument('--lr_decay', default=config['train']['lr_decay'], type=bool)
    parser.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=int)
    parser.add_argument('--early_stop', default=config['train']['early_stop'], type=bool)
    parser.add_argument('--dropout', default=config['train']['dropout'], type=float)
    parser.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    #log
    parser.add_argument('--log_step', type=int, default=config['log']['log_step'])
    parser.add_argument('--plot', type=str, default=config['log']['plot'])
    parser.add_argument('--log_dir', type=str, default='./')
    args = parser.parse_args()
    return args