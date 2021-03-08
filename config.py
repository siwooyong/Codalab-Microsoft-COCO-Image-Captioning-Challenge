# caption_config

class caption_config:
  num_workers = 16
  batch_size = hyper_parameters['batch_size']
  n_epochs = 50
  lr = hyper_parameters['lr']
  folder = hyper_parameters['save_address']
  verbose = True
  verbose_step = 1
  step_scheduler = False
  validation_scheduler = True
  SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
  scheduler_params = dict(
      mode = 'min',
      factor = 0.5,
      patience = 2,
      verbose = False, 
      threshold = 0.0001,
      threshold_mode = 'abs',
      cooldown = 0, 
      min_lr = 1e-8,
      eps = 1e-08)
