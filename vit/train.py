import os, sys

sys.path.append('../')
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import wandb
import warnings
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from metrics import classification_metrics
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from utils import num_workers
from Model import Model
from DataLoader import CocoCaptions, CocoCaptionsOnline
from pytorch_lightning import seed_everything
import time
from transformers import BertModel, BertTokenizer, BertConfig
from sklearn.metrics import roc_auc_score, accuracy_score

seed_everything(9)


class Train_Vit(pl.LightningModule):
    def __init__(self, batch_size=4, num_gpu=0, model_weight='../weights/bert_vit-model_16', weight_decay=0, lr=1e-3,
                 p_mask=0.15, img_mask=False, save_dir='models', size=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_gpu = num_gpu
        self.p_mask = p_mask
        self.lr = lr / self.num_gpu if self.num_gpu != 0 else lr
        self.weight_decay = weight_decay
        self.alpha = 1
        self.img_mask = img_mask
        self.tokenizer = BertTokenizer.from_pretrained('../weights/saved-bert-base-uncased')
        self.model = Model(model_weight)
        self.patches_size = self.model.bert.config.patches_size
        self.vocabulary = np.array(range(self.model.vocab_size))
        self.vocabulary = np.delete(self.vocabulary, [self.tokenizer.pad_token_id, self.tokenizer.mask_token_id,
                                                      self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        self.last_loss = 1e10
        self.log_step = 10
        self.save_dir = save_dir

        self.train_dataset = CocoCaptions(root='../../train2014',
                                          annFile='../../annotations/captions_train2014.json',
                                          transform=self.get_transform(224, True),
                                          end=size)
        self.val_dataset = CocoCaptions(root='../../val2014',
                                        annFile='../../annotations/captions_val2014.json',
                                        transform=self.get_transform(224, False),
                                        start=0, end=-25000)

        # self.train_dataset = CocoCaptionsOnline(annFile=path+'/annotations/captions_train2014.json',
        #                                 transform=transform)
        # self.val_dataset = CocoCaptionsOnline(annFile=path+'/annotations/captions_train2014.json',
        #                               transform=transform)

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cuda:0'))

    def get_transform(self, image_size, train):
        if train:
            bigger_image_size = (image_size // 8 + 1) * 8
            ts = [
                transforms.Resize((bigger_image_size, bigger_image_size)),
                transforms.RandomResizedCrop((image_size, image_size)),
                transforms.RandomHorizontalFlip()
            ]
        else:
            ts = [transforms.Resize((image_size, image_size))]
        ts += [
            transforms.ToTensor(),transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
        return transforms.Compose(ts)

    def preprocessing_data(self, img, cap):
        encoding = self.tokenizer(list(cap), return_tensors='pt', padding=True, truncation=True)
        tokens = encoding['input_ids']

        attention_mask = encoding['attention_mask']

        # mask tokens
        mask = np.random.choice([0, 1, 2, 3], tokens.shape,
                                p=[1 - self.p_mask, self.p_mask * 0.8, self.p_mask * 0.1, self.p_mask * 0.1])
        mask[np.where(attention_mask == 0)] = 0
        mask[np.where(tokens == self.tokenizer.sep_token_id)] = 0
        mask[np.where(tokens == self.tokenizer.cls_token_id)] = 0
        tokens[np.where(mask == 1)] = self.tokenizer.mask_token_id
        idx = np.where(mask == 2)
        tokens[idx] = torch.tensor(np.random.choice(self.vocabulary, tokens.shape), dtype=torch.long)[idx]

        # number of image embeddings
        num_image_embeddings = int(img.shape[2] * img.shape[3] / self.patches_size ** 2)
        attention_mask = self.get_attention_mask(attention_mask, num_image_embeddings)

        # mask images
        if self.img_mask:
            img_mask = np.random.choice([0, 1, 2],
                                        (img.shape[0], num_image_embeddings),
                                        p=[1 - self.p_mask, self.p_mask * 0.8, self.p_mask * 0.2])
        else:
            img_mask = None
        return tokens.to(img.device), mask, img_mask, attention_mask.to(img.device)

    def mix(self, cap):
        n = len(cap)

        idx_mix = np.arange(n // 2)
        mix_cap = cap[idx_mix]

        m = len(mix_cap) // 2

        cap[idx_mix] = np.concatenate((mix_cap[m:], mix_cap[:m]))

        y_bce = torch.cat((
            torch.zeros(n // 2),
            torch.ones(n // 2),
        ))

        part1 = torch.arange(start=m, end=n // 2, dtype=torch.int64)
        part2 = torch.arange(start=0, end=m, dtype=torch.int64)
        part3 = torch.arange(start=n // 2, end=n, dtype=torch.int64)
        y = torch.cat((part1, part2, part3))
        return cap, y_bce, y

    def get_attention_mask(self, attention_mask, image_embedding_size):
        ones = torch.ones((attention_mask.shape[0], image_embedding_size + 1)).long()
        attention_mask = torch.cat((attention_mask, ones), 1)
        return attention_mask

    def loss(self, y, y_scores):
        cross_entropy = nn.CrossEntropyLoss()
        loss_ti = cross_entropy(y_scores, y)
        loss_it = cross_entropy(y_scores.transpose(0, 1), y)

        return loss_ti, loss_it

    def bin_ce(self, input, target):
        sigmoid = nn.Sigmoid()
        loss = nn.BCELoss()
        return loss(sigmoid(input), target)

    def classification_metrics(self, y_true, y_score):
        sigmoid = nn.Sigmoid()
        y_score = sigmoid(y_score)

        # accuracy
        y_pred = (y_score > 0.5).float().detach().cpu().numpy()
        acc = accuracy_score(y_pred, y_true.cpu())

        # roc auc
        y_score = y_score.detach().cpu().numpy()
        roc_auc = roc_auc_score(y_true.cpu(), y_score)

        return acc, roc_auc

    def forward(self, batch):
        self.model.eval()
        img, cap = batch
        masked_cap, mask, img_mask, attention_mask = self.preprocessing_data(img, cap)
    
        y_scores = self.model(masked_cap, img, img_mask, attention_mask, 0)
        y_score = y_scores[np.arange(len(y_scores)), np.arange(len(y_scores))]
    
        sigmoid = nn.Sigmoid()
        y_score = sigmoid(y_score)
        y_pred = (y_score > 0.5).float().detach().cpu().numpy()
    
        return y_score, y_pred

    def step(self, batch):
        img, cap = batch

        cap, y_bce, y = self.mix(np.array(cap))
        y = y.to(img.device)
        y_bce = y_bce.to(img.device)

        masked_cap, mask, img_mask, attention_mask = self.preprocessing_data(img, cap)

        y_scores = self.model(masked_cap, img, img_mask, attention_mask, self.alpha)

        loss_ti, loss_it = self.loss(y, y_scores)
        loss = loss_ti + loss_it

        y_score = y_scores[np.arange(len(y_scores)), np.arange(len(y_scores))]
        bce = self.bin_ce(y_score, y_bce)

        loss += bce

        return loss, loss_ti, loss_it, bce, y_bce, y_score

    def training_step(self, batch, batch_idx):
        loss, loss_ti, loss_it, bce, y_bce, y_score = self.step(batch)
        return {"loss": loss, 'loss_ti': loss_ti, 'loss_it': loss_it, 'bce': bce, 'y_bce': y_bce, 'y_score': y_score}

    def training_step_end(self, batch_parts):
        n = self.num_gpu
        if isinstance(batch_parts, dict):
            batch_parts = [batch_parts]
            n = 1
        loss, loss_ti, loss_it, bce = 0.0, 0.0, 0.0, 0.0
        y_bce, y_score = torch.tensor([]), torch.tensor([])
        for i in range(n):
            loss += batch_parts[i]['loss']
            loss_ti += batch_parts[i]['loss_ti']
            loss_it += batch_parts[i]['loss_it']
            bce += batch_parts[i]['bce']
            y_bce = torch.cat((y_bce.to(batch_parts[i]['y_bce'].device), batch_parts[i]['y_bce']))
            y_score = torch.cat((y_score.to(batch_parts[i]['y_score'].device), batch_parts[i]['y_score']))

        if loss.item()/n < 1:
            self.alpha = 0.9 * self.alpha

        if self.global_step % self.log_step == 0:
            self.logger.experiment.log({"train/epoch": self.current_epoch,
                                        "train/loss": loss / n,
                                        "train/loss_ti": loss_ti / n,
                                        "train/loss_it": loss_it / n,
                                        "train/bce": bce / n,
                                        "train/alpha": self.alpha}, step=self.global_step)

        return {"loss": loss, 'loss_ti': loss_ti, 'loss_it': loss_it, 'y_bce': y_bce, 'y_score': y_score}

    def training_epoch_end(self, training_step_outputs):
        loss, loss_ti, loss_it = 0.0, 0.0, 0.0
        y_bce, y_score = torch.tensor([]), torch.tensor([])
        n = len(training_step_outputs)
        for out in training_step_outputs:
            loss += out['loss']
            loss_ti += out['loss_ti']
            loss_it += out['loss_it']
            y_bce = torch.cat((y_bce.to(out['y_bce'].device), out['y_bce']))
            y_score = torch.cat((y_score.to(out['y_score'].device), out['y_score']))
        acc, roc_auc = self.classification_metrics(y_bce, y_score)
        self.logger.experiment.log({"train_metrics/acc": acc,
                                    "train_metrics/roc_auc": roc_auc}, step=self.global_step)
        print("Epoch: {}, Loss: {:.5}, loss_ti: {:.5}, loss_it: {:.5}".format(
            self.current_epoch, loss / n, loss_ti / n, loss_it / n))

    def validation_step(self, batch, batch_idx):
        loss, loss_ti, loss_it, bce, y_bce, y_score = self.step(batch)
        return {"loss": loss, 'loss_ti': loss_ti, 'loss_it': loss_it, 'bce': bce, 'y_bce': y_bce, 'y_score': y_score}

    def validation_step_end(self, batch_parts):
        n = self.num_gpu
        if isinstance(batch_parts, dict):
            batch_parts = [batch_parts]
            n = 1
        loss, loss_ti, loss_it, bce = 0.0, 0.0, 0.0, 0.0
        y_bce, y_score = torch.tensor([]), torch.tensor([])
        for i in range(n):
            loss += batch_parts[i]['loss']
            loss_ti += batch_parts[i]['loss_ti']
            loss_it += batch_parts[i]['loss_it']
            bce += batch_parts[i]['bce']
            y_bce = torch.cat((y_bce.to(batch_parts[i]['y_bce'].device), batch_parts[i]['y_bce']))
            y_score = torch.cat((y_score.to(batch_parts[i]['y_score'].device), batch_parts[i]['y_score']))

        return {"loss": loss / n, 'loss_ti': loss_ti / n, 'loss_it': loss_it / n, 'bce': bce / n, 'y_bce': y_bce,
                'y_score': y_score}

    def validation_epoch_end(self, validation_step_outputs):
        loss, loss_ti, loss_it, bce = 0.0, 0.0, 0.0, 0.0
        y_bce, y_score = torch.tensor([]), torch.tensor([])
        n = len(validation_step_outputs)
        for out in validation_step_outputs:
            loss += out['loss']
            loss_ti += out['loss_ti']
            loss_it += out['loss_it']
            bce += out['bce']
            y_bce = torch.cat((y_bce.to(out['y_bce'].device), out['y_bce']))
            y_score = torch.cat((y_score.to(out['y_score'].device), out['y_score']))
        acc, roc_auc = self.classification_metrics(y_bce, y_score)
        print(
            "VALIDATION! Epoch: {}, Loss: {:.5}, loss_ti: {:.5}, loss_it: {:.5}, bce: {:.5}, Acc: {:.5}, ROC AUC: {:.5}".format(
                self.current_epoch, loss.detach().item() / n, loss_ti.detach().item() / n,
                                    loss_it.detach().item() / n, bce.detach().item() / n,
                acc, roc_auc))
        epoch = self.current_epoch if self.global_step < 1 else self.current_epoch + 1
        self.logger.experiment.log({"epoch": epoch,
                                    "val/loss": loss / n,
                                    'val/loss_ti': loss_ti / n,
                                    'val/loss_it': loss_it / n,
                                    'val/bce': bce / n}, step=self.global_step)

        self.logger.experiment.log(
            {"val_metrics/accuracy": acc, 'val_metrics/roc_auc': roc_auc},
            step=self.global_step)

        self.save(f'exp/{self.save_dir}/{epoch}_vit.pkl')

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers, drop_last=True)

    def save(self, path):
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    print("START")
    os.environ['WANDB_MODE'] = "dryrun"

    parser = argparse.ArgumentParser(description='ViT')
    parser.add_argument('--num_gpu', type=int, default=2, help='number of gpu (default: 2)')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size multiple of four')
    parser.add_argument('--model_weights', type=str, default='../weights/bert_vit-model_16_imagenet')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--p_mask', type=float, default=0)
    parser.add_argument('--img_mask', type=bool, default=False)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--size', default=None)

    arg = parser.parse_args()

    hyperparameters = {
        'num_gpu': arg.num_gpu,
        'batch_size': arg.batch_size,
        'model_weight': arg.model_weights,
        'weight_decay': arg.weight_decay,
        'lr': arg.lr,
        'p_mask': arg.p_mask,
        'img_mask': arg.img_mask,
        'save_dir': arg.save_dir,
        'size': arg.size
    }
    print(hyperparameters)

    wandb_logger = WandbLogger(name="gpu_vit", project="pytorch-vit", offline=True, config=hyperparameters)
    train_vit = Train_Vit(**hyperparameters)
    wandb_logger.watch(train_vit, log='all')
    trainer = pl.Trainer(gpus=hyperparameters['num_gpu'], max_epochs=10 ** 10, min_epochs=10 ** 10, accelerator='ddp',
                         logger=wandb_logger)
    trainer.fit(train_vit)
