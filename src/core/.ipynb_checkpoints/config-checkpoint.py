class Config:
    LOG_DIR = None
    USE_AMP = True
    ENABLE_GRADSCALER = True
    NUMS_STEP = 20000
    CUDNN_BENCHMARK = False
    SEED = 1273
    VALIDATION_STEP = 2
    BATCH_SIZE = 6
    VAL_EPOCH_EVERY_N_TRAIN_STEP = 200
    ACCUMULATED_GRADIENT_STEPS = 1
    