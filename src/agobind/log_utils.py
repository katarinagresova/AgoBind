import comet_ml

def log_extra(config, f1_score, recall, precision):
    current_experiment = comet_ml.get_global_experiment()
    afterlog_experiment = comet_ml.ExistingExperiment(previous_experiment=current_experiment.get_key())
    exp_name = str(config).replace(' ','')
    afterlog_experiment.set_name(exp_name)
    afterlog_experiment.log_parameters(config)
    afterlog_experiment.log_metric("test F1 score", f1_score)
    afterlog_experiment.log_curve(f"pr-curve", recall, precision)
    afterlog_experiment.log_model("DNABERT_CLASH", "./best_model")
    afterlog_experiment.end()

