from typing import Optional
import numpy as np
import torch

nota = "NOTA"


class SpecialLoss:
    def __init__(self) -> None:
        self.total_loss = 0.
        self.count = 0.

    def __call__(self,loss_term):
        self.total_loss += loss_term.item()
        self.count += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated accuracy.
        """
        if self.count > 1e-12:
            accuracy = float(self.total_loss) / float(self.count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy

    def reset(self):
        self.total_loss = 0.0
        self.count = 0.0

class F1AllClasses():
    def __init__(self,
                 beta: float = 1.0,
                 average: str = None) -> None:
        self.beta = beta
        self.average = average
        self.nota_class = "NOTA"
        self.labels_dict = {}
        self.labels_dict[self.nota_class] = 0
        self.num_training_categories = 64 + 1
        self.num_development_categories = 6 + 1
        self.num_testing_categories = 10 + 1


    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 user_interest: list,
                 is_training: bool,
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        """
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions, gold_labels, mask)

        # add "NOTA" as lat class
        for u in user_interest:
            u.append(self.nota_class)
        # Calculate true_positive_sum, true_negative_sum, pred_sum, true_sum

        # if (gold_labels >= num_classes).any():
        #     raise ConfigurationError("A gold label passed to FBetaMeasure contains "
        #                              f"an id >= {num_classes}, the number of classes.")

        # It means we call this metric at the first time
        # when `self._true_positive_sum` is None.
        num_classes = self.num_training_categories


        if self._true_positive_sum is None:
            self._true_positive_sum = torch.zeros(num_classes)
            self._true_sum = torch.zeros(num_classes)
            self._pred_sum = torch.zeros(num_classes)
            self._total_sum = torch.zeros(num_classes)

        if mask is None:
            mask = torch.ones(gold_labels.size(0)*gold_labels.size(1))

        mask = mask.to(dtype=torch.bool)
        gold_labels = torch.flatten(gold_labels).float()
        class_indecies = self.convert_indication_to_class_index(user_interest)
        the_gold_class_list_this_batch = self.labels_to_classes_index(gold_labels.long(), class_indecies)

        argmax_predictions = predictions.max(dim=-1)[1].float()
        argmax_predictions = argmax_predictions.view(argmax_predictions.size(0) * argmax_predictions.size(1))
        the_pred_list = self.labels_to_classes_index(argmax_predictions.long(), class_indecies)

        true_positives = (the_gold_class_list_this_batch == the_pred_list) * mask
        true_positives_bins = the_gold_class_list_this_batch[true_positives]

        # num_classes = len(self.labels_dict)
        # Watch it:
        # The total numbers of true positives under all _predicted_ classes are zeros.
        if true_positives_bins.shape[0] == 0:
            true_positive_sum = torch.zeros(num_classes)
        else:
            true_positive_sum = torch.bincount(true_positives_bins.long(), minlength=num_classes).float()

        pred_bins = the_pred_list[mask].long()
        # Watch it:
        # When the `mask` is all 0, we will get an _empty_ tensor.
        if pred_bins.shape[0] != 0:
            pred_sum = torch.bincount(pred_bins, minlength=num_classes).float()
        else:
            pred_sum = torch.zeros(num_classes)

        gold_labels_bins = the_gold_class_list_this_batch[mask].long()
        if the_gold_class_list_this_batch.shape[0] != 0:
            true_sum = torch.bincount(gold_labels_bins, minlength=num_classes).float()
        else:
            true_sum = torch.zeros(num_classes)

        # new_known_class = num_classes - self._true_positive_sum.size(0)
        # assert new_known_class >= 0
        # if new_known_class > 0:
        #     concat_to_the_end = torch.zeros(new_known_class)
        #     self._true_positive_sum = torch.cat((self._true_positive_sum,concat_to_the_end))
        #     self._pred_sum = torch.cat((self._pred_sum,concat_to_the_end))
        #     self._true_sum = torch.cat((self._true_sum,concat_to_the_end))
        #
        if self._true_positive_sum.size(0) != true_positive_sum.size(0):
            print("true_positive_sum size:", true_positive_sum.size())
            print("self.true_positive_sum size:",self._true_positive_sum.size())
            print("num classes:",num_classes)
            # print("new_known_class:",new_known_class)

        self._true_positive_sum += true_positive_sum
        self._pred_sum += pred_sum
        self._true_sum += true_sum
        self._total_sum += mask.sum().to(torch.float)

    def convert_indication_to_class_index(self,ui):
        list_of_classes = []
        if type(ui[0]) is list:
            ui_single_ = list(np.concatenate(ui))
        for c in ui_single_:
            if c not in self.labels_dict:
                self.labels_dict[c] = len(self.labels_dict)
            list_of_classes.append(self.labels_dict[c])
        return torch.tensor(list_of_classes).long()


    def labels_to_classes_index(self, labels, user_interest):
        x = user_interest[labels]
        return x

    def get_metric(self,
                   reset: bool = False):
        """
        Returns
        -------
        A tuple of the following metrics based on the accumulated count statistics:
        precisions : List[float]
        recalls : List[float]
        f1-measures : List[float]

        If ``self.average`` is not ``None``, you will get ``float`` instead of ``List[float]``.
        """
        # self._labels = list(filter(lambda x: x > 0, self.labels_dict.values()))
        if self._true_positive_sum is None:
            raise RuntimeError("You never call this metric before.")

        tp_sum = self._true_positive_sum
        pred_sum = self._pred_sum
        true_sum = self._true_sum

        # remove NOTA scores
        tp_sum = tp_sum[1:]
        pred_sum = pred_sum[1:]
        true_sum = true_sum[1:]

        tp_sum = tp_sum.sum()
        pred_sum = pred_sum.sum()
        true_sum = true_sum.sum()

        beta2 = self._beta ** 2
        # Finally, we have all our sufficient statistics.
        precision = _prf_divide(tp_sum, pred_sum)
        recall = _prf_divide(tp_sum, true_sum)
        fscore = ((1 + beta2) * precision * recall /
                  (beta2 * precision + recall))

        # precision = precision[(true_sum + pred_sum) != 0]
        # recall = recall[(true_sum + pred_sum) != 0]

        fscore[tp_sum == 0] = 0.0
        # fscore = fscore[(true_sum + pred_sum) != 0]

        if reset:
            self.reset()

        # Retain only selected labels and order them

        return precision.item(), recall.item(), fscore.item()

    def reset(self) -> None:
        super().reset()
        self.labels_dict = {self.nota_class: 0}
        # print("RESET WAS DONE")


def _prf_divide(numerator, denominator):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements to zero.
    """
    result = numerator / denominator
    mask = denominator == 0.0
    if not mask.any():
        return result

    # remove nan
    result[mask] = 0.0
    return result
