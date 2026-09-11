from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class TrainingMetricsHook(HookBase):

    def __init__(
        self,
        output_dir="training_process",
        metrics_file="epoch_metrics.txt",
        best_file="best_metrics.txt",
    ):
        self.output_dir = output_dir
        self.metrics_file = metrics_file
        self.best_file = best_file

    def before_train(self):
        if not comm.is_main_process():
            return

        self.save_dir = Path(self.trainer.cfg.save_path) / self.output_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.save_dir / self.metrics_file
        self.best_path = self.save_dir / self.best_file
        self.metrics_curve_path = self.save_dir / "metrics_curve.png"
        self.loss_lr_curve_path = self.save_dir / "loss_lr_curve.png"

        resume = bool(getattr(self.trainer.cfg, "resume", False))

        if not resume:
            for path in [
                self.metrics_path,
                self.best_path,
                self.metrics_curve_path,
                self.loss_lr_curve_path,
            ]:
                if path.exists():
                    path.unlink()

    def after_epoch(self):
        if not bool(getattr(self.trainer.cfg, "evaluate", False)):
            return

        if not comm.is_main_process():
            return

        storage = self.trainer.storage
        epoch = self.trainer.epoch + 1
        max_epoch = self.trainer.max_epoch

        train_loss = float(storage.history("loss").avg)
        val_loss = float(storage.history("val_loss").avg)

        intersection = np.asarray(
            storage.history("val_intersection").total,
            dtype=np.float64,
        )
        union = np.asarray(
            storage.history("val_union").total,
            dtype=np.float64,
        )
        target = np.asarray(
            storage.history("val_target").total,
            dtype=np.float64,
        )

        eps = 1e-10
        output_area = union + intersection - target

        iou = intersection / (union + eps)
        recall = intersection / (target + eps)
        precision = intersection / (output_area + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)

        mIoU = float(np.mean(iou))
        macroF1 = float(np.mean(f1))
        mAcc = float(np.mean(recall))
        allAcc = float(
            np.sum(intersection) / (np.sum(target) + eps)
        )

        learning_rates = [
            float(group["lr"])
            for group in self.trainer.optimizer.param_groups
        ]
        lr = max(learning_rates)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "mIoU": mIoU,
            "macroF1": macroF1,
            "mAcc": mAcc,
            "allAcc": allAcc,
        }

        self._append_epoch_metrics(row)

        previous_best = getattr(
            self.trainer,
            "best_metric_value",
            0.0,
        )

        if previous_best is None:
            previous_best = 0.0

        if hasattr(previous_best, "item"):
            previous_best = float(previous_best.item())
        else:
            previous_best = float(previous_best)

        is_new_best = mIoU > previous_best
        class_names = self.trainer.cfg.data.names

        if is_new_best:
            self._write_best_metrics(
                row=row,
                class_names=class_names,
                iou=iou,
                f1=f1,
                precision=precision,
                recall=recall,
            )
            best_miou = mIoU
            best_flag = "  <<< NEW BEST"
        else:
            best_miou = previous_best
            best_flag = ""

        separator = "=" * 100

        self.trainer.logger.info(separator)

        self.trainer.logger.info(
            f"Epoch [{epoch}/{max_epoch}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"mIoU: {mIoU:.4f} | "
            f"Macro F1: {macroF1:.4f} | "
            f"mAcc: {mAcc:.4f} | "
            f"LR: {lr:.8f}"
        )

        self.trainer.logger.info(
            f"Best mIoU: {best_miou:.4f}{best_flag}"
        )

        self.trainer.logger.info(separator)

    def _append_epoch_metrics(self, row):
        file_exists = self.metrics_path.exists()

        with open(
            self.metrics_path,
            "a",
            encoding="utf-8",
        ) as f:
            if not file_exists:
                f.write("\t".join(row.keys()) + "\n")

            values = []
            for value in row.values():
                if isinstance(value, float):
                    values.append(f"{value:.6f}")
                else:
                    values.append(str(value))

            f.write("\t".join(values) + "\n")

    def _write_best_metrics(
        self,
        row,
        class_names,
        iou,
        f1,
        precision,
        recall,
    ):
        with open(
            self.best_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write("=" * 60 + "\n")
            f.write("PTv3 Best Validation Result\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Best Epoch    : {row['epoch']}\n")
            f.write(f"Best mIoU     : {row['mIoU']:.6f}\n")
            f.write(f"Macro F1      : {row['macroF1']:.6f}\n")
            f.write(f"mAcc          : {row['mAcc']:.6f}\n")
            f.write(f"Train Loss    : {row['train_loss']:.6f}\n")
            f.write(f"Val Loss      : {row['val_loss']:.6f}\n")
            f.write(f"Learning Rate : {row['lr']:.8f}\n")

            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write("Per-Class Metrics\n")
            f.write("=" * 60 + "\n")
            f.write("Class\tIoU\tF1\tPrecision\tRecall\n")

            for i, name in enumerate(class_names):
                f.write(
                    f"{name}\t"
                    f"{iou[i]:.6f}\t"
                    f"{f1[i]:.6f}\t"
                    f"{precision[i]:.6f}\t"
                    f"{recall[i]:.6f}\n"
                )

    def _read_metrics(self):
        with open(
            self.metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(lines) <= 1:
            return []

        header = lines[0].split("\t")
        rows = []

        for line in lines[1:]:
            values = line.split("\t")
            rows.append(dict(zip(header, values)))

        return rows

    def _plot_curves(self):
        rows = self._read_metrics()

        if len(rows) == 0:
            return

        epochs = [
            int(row["epoch"])
            for row in rows
        ]

        metric_names = [
            "mIoU",
            "macroF1",
            "mAcc",
        ]

        plt.figure(figsize=(10, 6))

        for metric in metric_names:
            values = [
                float(row[metric])
                for row in rows
            ]
            plt.plot(
                epochs,
                values,
                label=metric,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("PTv3 Validation Metrics")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            self.metrics_curve_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        train_loss = [
            float(row["train_loss"])
            for row in rows
        ]
        val_loss = [
            float(row["val_loss"])
            for row in rows
        ]
        lr = [
            float(row["lr"])
            for row in rows
        ]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        line1 = ax1.plot(
            epochs,
            train_loss,
            label="Train Loss",
        )
        line2 = ax1.plot(
            epochs,
            val_loss,
            label="Val Loss",
        )

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()

        line3 = ax2.plot(
            epochs,
            lr,
            linestyle="--",
            label="Learning Rate",
        )

        ax2.set_ylabel("Learning Rate")

        lines = line1 + line2 + line3
        labels = [
            line.get_label()
            for line in lines
        ]

        ax1.legend(
            lines,
            labels,
            loc="best",
        )

        plt.title("PTv3 Loss and Learning Rate")
        fig.tight_layout()

        plt.savefig(
            self.loss_lr_curve_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def after_train(self):
        if not comm.is_main_process():
            return

        if not self.metrics_path.exists():
            return

        self._plot_curves()

        self.trainer.logger.info(
            f"Training process saved to: {self.save_dir}"
        )

        if self.best_path.exists():
            self.trainer.logger.info(
                f"Best result saved to: {self.best_path}"
            )
