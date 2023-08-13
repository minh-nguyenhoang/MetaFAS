from tqdm.auto import tqdm
import torch
from torchmetrics.classification.roc import ROC
from torchmetrics.classification.confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader):
    device = next(model.parameters()).device
    pr_list = []
    gt_list = []
    cm = ConfusionMatrix(num_classes=2)
    auc = ROC()
    for step, batch in tqdm(dataloader):
        img, map_gt, label_gt = batch[0].float().to(device), batch[1].float().to(device), batch[2].long().to(device)

        label_pr, map_pr, _ = model(img)

        pr_list.append(label_pr.detach().cpu())
        gt_list.append(label_gt.detach().cpu())

    pr_list = torch.cat(pr_list)[:,1]
    gt_list = torch.cat(gt_list)

    cm_out = cm(pr_list, gt_list)
    


    fpr, tpr, thresholds = auc(pr_list, gt_list)
    plt.plot(fpr,fpr, 'r--')
    plt.plot(fpr,tpr)
    plt.title('ROC curve')
    plt.show()

    return 



