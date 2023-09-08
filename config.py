class Config:

    def __init__(self):

        self.batch_size = 1

        self.n_epochs = 20

        self.lr = 0.0001
        self.warmup_steps = 0

        self.optimizer = 'adamw_torch'

        self.fp16 = True  # Mixed Precision Training

        self.data_dir = 'Data'
        self.save_path = 'saved_data'
        self.model_save_path = 'saved_data/gtzan_music'

        self.run_name = 'wave2vec2'

        self.drop_last = False
        self.num_workers = 0
        self.pin_memory = True

        self.eval_steps = 500
        self.save_steps = 100
        self.logging_step = 10

        self.seed = 42
