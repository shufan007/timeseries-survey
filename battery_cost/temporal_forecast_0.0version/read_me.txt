introduction:
    this work try to resolve a temporal forecasting problem, and dataset is composed by three part: h17_seq_features

requirements:
    pytorch

directory instructions:
    ./model: where storing model framework and its related functions;
    ./lib: where storing dataloader, config handler, logger, metrics and training initialization;
    ./checkpoints: where to deposit experiments results;
    ./runner: including training structure and the config of input;

Training:
    you can run this work by using: python run.py