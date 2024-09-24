import torch
import torch.nn.functional as F


class LOSSMIA:
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def model_eval(self, model, data_samples, epsilon=10e-9):
        with torch.no_grad():
            model = model.to(self.device)
            data_samples = {k: v.to(self.device) for k, v in data_samples.items()}
            input_ids = data_samples["input_ids"]
            attention_mask = data_samples["attention_mask"]
            targets = data_samples["labels"]
            outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
            logits = outputs.logits
            loss_val = (
                F.cross_entropy(logits, targets, reduction="none").detach().clone()
            )

            nan_mask = torch.isnan(loss_val)
            loss_val[nan_mask] = torch.tensor(1 / epsilon).to(self.device)
            inf_mask = torch.isinf(loss_val)
            loss_val[inf_mask] = torch.tensor(1 / epsilon).to(self.device)
            return loss_val

    def attack_dataset(
        self,
        victim_model,
        in_dataloader,
        out_dataloader,
        in_size=32299,
        out_size=8484,
        epsilon=10e-9,
    ):
        victim_model.eval()
        loss_vals = {
            "in": torch.zeros((in_size,), dtype=torch.float32, device=self.device),
            "out": torch.zeros((out_size,), dtype=torch.float32, device=self.device),
        }
        with torch.no_grad():
            last = 0
            for data_samples in in_dataloader:
                loss_in = -self.model_eval(victim_model, data_samples, epsilon=epsilon)
                loss_vals["in"][last : last + len(data_samples["labels"])] = loss_in
                last += len(data_samples["labels"])
            loss_vals["in"] = loss_vals["in"][:last].cpu()

            last = 0
            for data_samples in out_dataloader:
                loss_out = -self.model_eval(victim_model, data_samples, epsilon=epsilon)
                loss_vals["out"][last : last + len(data_samples["labels"])] = loss_out
                last += len(data_samples["labels"])
            loss_vals["out"] = loss_vals["out"][:last].cpu()
            return loss_vals
