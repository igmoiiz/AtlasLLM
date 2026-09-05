"""Memorization metric: train vs held-out passage loss gap.

A healthy model generalizes: predictions on held-out passages cost nearly as
much as on training passages. A memorizing model has flashed the training set
so many times that train-passage loss collapses far below held-out loss. We
measure both with :func:`evaluation.perplexity.evaluate_loss` (identical loss
and loader contract as training/validation).
"""


import torch

from evaluation.perplexity import evaluate_loss, perplexity


def memorize_gap(train_loss: float, heldout_loss: float) -> float:
    """Sign of memorization: held-out loss above train loss (in nats)."""
    return heldout_loss - train_loss


def evaluate_memorization(
    model: torch.nn.Module,
    train_loader,
    heldout_loader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Measure the train-vs-held-out loss gap on matching passages.

    Both loaders must slice the *same context length* so the comparison is
    per-token; ``train_loader`` should be drawn from the training split and
    ``heldout_loader`` from the validation/test split.
    """
    train = evaluate_loss(model, train_loader, device, max_batches)
    heldout = evaluate_loss(model, heldout_loader, device, max_batches)
    return {
        "train_loss": train,
        "heldout_loss": heldout,
        "gap": memorize_gap(train, heldout),
        "train_perplexity": perplexity(train),
        "heldout_perplexity": perplexity(heldout),
    }
