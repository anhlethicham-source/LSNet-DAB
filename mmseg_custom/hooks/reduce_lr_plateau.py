from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ReduceLROnPlateauHook(Hook):
    """Giảm LR ×factor khi metric không cải thiện sau patience lần eval liên tiếp."""

    def __init__(self, monitor='Dice', factor=0.5, patience=5, min_lr=1e-6, eval_interval=4000):
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.eval_interval = eval_interval
        self.best = None
        self.wait = 0

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.eval_interval != 0:
            return

        val = runner.log_buffer.output.get(self.monitor)
        if val is None:
            return

        if self.best is None or val > self.best:
            self.best = val
            self.wait = 0
            runner.logger.info(f'ReduceLR: {self.monitor}={val:.4f} improved → best={self.best:.4f}')
        else:
            self.wait += 1
            runner.logger.info(
                f'ReduceLR: {self.monitor}={val:.4f} no improvement ({self.wait}/{self.patience})'
            )
            if self.wait >= self.patience:
                for pg in runner.optimizer.param_groups:
                    old_lr = pg['lr']
                    pg['lr'] = max(old_lr * self.factor, self.min_lr)
                runner.logger.info(
                    f'ReduceLR: lr {old_lr:.2e} → {pg["lr"]:.2e}'
                )
                self.wait = 0
