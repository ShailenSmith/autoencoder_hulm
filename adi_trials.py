def objective(trial: optuna.trial.Trial, args, dataloaderModule):
    # Set the seed for reproducibility. Model init with same weight for every trial
    pl.seed_everything(args.seed)
    
    args.lr = trial.suggest_float("lr", 9e-5, 1e-2, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    args.output_dropout = trial.suggest_float("output_dropout", 0.0, 0.5)
    args.hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
    
    lightning_module = MILightningModule(args)

    trial_number = trial.number
    trial_exp_name = args.experiment_name+'_trial_{}'.format(trial_number)

    # Get the logger and log the hyperparameters
    logger = get_logger('comet', workspace=args.workspace, project_name=args.project_name, experiment_name=trial_exp_name, save_dir=args.output_dir)
    logger.log_hyperparams(args.__dict__)
    
    # add callback for early stopping
    callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")]
    if args.early_stopping_patience>0: callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience, 
                                                                             mode=args.early_stopping_mode, min_delta=args.early_stopping_min_delta)]
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir=args.output_dir, logger=logger,
                        callbacks=callbacks, min_epochs=args.min_epochs, max_epochs=args.epochs, enable_checkpointing=False)
    trainer.fit(lightning_module, train_dataloaders=dataloaderModule.train_dataloader(), val_dataloaders=dataloaderModule.val_dataloader())
    
    score = np_min(lightning_module.epoch_loss['val'])
    logger.experiment.end()
    
    SRC_PATH = os.path.join(args.output_dir, '{}/{}/'.format(logger._project_name, logger._experiment_key))
    DEST_PATH = args.HPARAM_OUTPUT_DIR + '/{}/'.format(logger._experiment_key)
    print ('Moving {} to {}'.format(SRC_PATH, DEST_PATH))
    if os.path.exists(SRC_PATH): shutil.move(SRC_PATH, DEST_PATH)