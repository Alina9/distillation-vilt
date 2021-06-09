import os, sys

sys.path.append('../')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import wandb
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizer
from bert import Bert
from image_bert import Vit
from vit.DataLoader import CocoCaptions, CocoCaptionsOnline
from pytorch_lightning.metrics.functional import accuracy
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

seed_everything(9)
num_workers = 16


class Train_Distillation(pl.LightningModule):
    def __init__(self, num_gpu, batch_size, vit_weights='../weights/vit-model_16',
                 student_weights='../weights/bert_vit-model_16', data='paired', vit_layers=None, weight_decay=0,
                 lr=1e-3, alpha=1, a=1, vit_coeff=1, bert_coeff=1, p_mask=0.15, masked=False, vit_head=False,
                 save_dir='models'):
        super().__init__()

        self.num_gpu = num_gpu
        self.batch_size = batch_size
        self.data = data
        self.vit_head = vit_head
        self.weight_decay = weight_decay
        self.lr = lr / self.num_gpu if self.num_gpu != 0 else lr
        self.bert_model = Bert('../weights/saved-bert-with-head')
        self.vit_model = Vit(vit_weights, freezed=True)
        self.student_model = Vit(student_weights)  # Vit(None)
        self.patches_size = self.vit_model.vit_model.config.patches_size
        self.tokenizer = BertTokenizer.from_pretrained('../weights/saved-bert-base-uncased')
        self.vocabulary = np.array(range(self.bert_model.vocab_size))
        self.vocabulary = np.delete(self.vocabulary, [self.tokenizer.pad_token_id, self.tokenizer.mask_token_id,
                                                      self.tokenizer.unk_token_id, self.tokenizer.sep_token_id])
        self.bert_model.eval()
        self.vit_model.eval()

        self.alpha = alpha
        self.vit_coeff = vit_coeff
        self.bert_coeff = bert_coeff
        self.p_mask = p_mask
        self.masked = masked
        self.log_step = 100
        self.save_dir = save_dir
        self.a = a

        if vit_layers is None:
            self.vit_layers = [i for i in range(13)]
        else:
            self.vit_layers = vit_layers

        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.train_dataset = CocoCaptions(root='../../train2014',
                                          annFile='../../annotations/captions_train2014.json',
                                          transform=transform)
        self.val_dataset = CocoCaptions(root='../../val2014',
                                        annFile='../../annotations/captions_val2014.json',
                                        transform=transform)

        # self.train_dataset = CocoCaptionsOnline(annFile='../annotations/captions_train2014.json',
        #                                 transform=transform)
        # self.val_dataset = CocoCaptionsOnline(annFile='../annotations/captions_val2014.json',
        #                               transform=transform)

    def preprocessing_data(self, img, cap):
        encoding = self.tokenizer(list(cap), return_tensors='pt', padding=True, truncation=True)
        tokens = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        # mask tokens
        if self.masked:
            mask = np.zeros(tokens.shape)
            while mask.sum() == 0:
                mask = np.random.choice([0, 1, 2, 3], tokens.shape,
                                        p=[1 - self.p_mask, self.p_mask * 0.8, self.p_mask * 0.1, self.p_mask * 0.1])
                mask[np.where(attention_mask == 0)] = 0
                mask[np.where(tokens == self.tokenizer.sep_token_id)] = 0
                mask[np.where(tokens == self.tokenizer.cls_token_id)] = 0
                tokens[np.where(mask == 1)] = self.tokenizer.mask_token_id
                idx = np.where(mask == 2)
                tokens[idx] = torch.tensor(np.random.choice(self.vocabulary, tokens.shape), dtype=torch.long)[idx]
            return img, tokens.to(img.device), attention_mask.to(img.device), mask

        return img, tokens.to(img.device), attention_mask.to(img.device), None

    def mix(self, cap):
        if self.data == 'paired':
            return cap
        n = len(cap)

        if self.data == 'not paired':
            k = n
        else:
            k = n // 2

        idx_mix = np.arange(k)
        mix_cap = cap[idx_mix]

        m = len(mix_cap) // 2

        cap[idx_mix] = np.concatenate((mix_cap[m:], mix_cap[:m]))

        return cap

    def get_attention_mask(self, attention_mask, num_image_embeddings):
        ones = torch.ones((attention_mask.shape[0], num_image_embeddings)).long().to(attention_mask.device)
        attention_mask = torch.cat((attention_mask, ones), 1)
        return attention_mask

    def metric(self, y_score, y_true):
        softmax = nn.Softmax(dim=1)
        y_score = softmax(y_score)

        # accuracy
        y_pred = y_score.max(1)[1]
        acc = accuracy(y_pred, y_true)
        return acc

    def head_cross_entropy(self, distr, target_distr):
        logsoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)
        return torch.mean(-torch.sum(softmax(target_distr) * logsoftmax(distr), 1))

    def distillation_loss(self, bert_output, vit_hidden_states, student_output, student_hidden_states,
                          student_logits=None, vit_logits=None, mask=None, cap=None):
        mse = nn.MSELoss()
        cross_entropy = nn.CrossEntropyLoss()

        vit_len = vit_hidden_states.shape[2]
        vit_hidden_states = vit_hidden_states[self.vit_layers, :, 1:]  ## without cls

        num_hidden_states = vit_hidden_states.shape[0]

        student_vit_hidden_states = student_hidden_states[self.vit_layers, :, -vit_len + 1:]  ## without cls
        student_prediction_logits, _ = self.bert_model.bert.cls(student_output[0][:, :-vit_len], student_output[1])

        # bert loss
        # bert loss for distillation
        bert_distillation_loss = mse(student_prediction_logits, bert_output)

        # bert loss for prediction tokens
        if self.masked:
            idx = np.where(mask != 0)
            encoding = self.tokenizer(list(cap), return_tensors='pt', padding=True, truncation=True)
            tokens = encoding['input_ids']
            bert_pred_loss = cross_entropy(student_prediction_logits[idx],
                                           tokens[idx].to(student_prediction_logits.device)) if len(
                np.array(idx)[0]) > 0 else student_prediction_logits.sum() * 0
            acc = self.metric(student_prediction_logits[idx], tokens[idx].to(student_prediction_logits.device))

            org_bert_pred_loss = cross_entropy(bert_output[idx],
                                               tokens[idx].to(bert_output.device)) if len(
                np.array(idx)[0]) > 0 else bert_output.sum() * 0
            org_acc = self.metric(bert_output[idx], tokens[idx].to(bert_output.device))

            # total bert loss
            total_bert_loss = self.alpha * bert_distillation_loss + (1 - self.alpha) * bert_pred_loss
        else:
            total_bert_loss = bert_distillation_loss
            bert_pred_loss, acc = 0, 0
            org_acc, org_bert_pred_loss = 0.0, 0.0

        # vit loss
        if self.vit_head:
            head_vit_loss = self.head_cross_entropy(student_logits, vit_logits)
        else:
            head_vit_loss = 0

        if self.vit_layers is not None:
            vit_loss = np.array(
                [mse(student_vit_hidden_states[i], vit_hidden_states[i]).detach().item() for i in
                 range(num_hidden_states)])
            layers_vit_loss = mse(student_vit_hidden_states, vit_hidden_states)
        else:
            layers_vit_loss = 0
            vit_loss = 0

        total_vit_loss = self.vit_coeff * layers_vit_loss + head_vit_loss

        loss = total_vit_loss + self.bert_coeff * total_bert_loss
        return loss, bert_distillation_loss, bert_pred_loss, vit_loss, head_vit_loss, acc, org_acc, org_bert_pred_loss

    def step(self, batch):
        img, cap = batch

        cap = self.mix(np.array(cap))
        img, tokens, attention_mask, mask = self.preprocessing_data(img, cap)

        bert_prediction_logits, _ = self.bert_model(tokens, attention_mask)
        vit_logits, vit_hidden_states = self.vit_model(img)

        # number of image embeddings
        num_image_embeddings = int(img.shape[2] * img.shape[3] / self.patches_size ** 2)
        attention_mask = self.get_attention_mask(attention_mask, num_image_embeddings + 1)

        student_logits, student_output, student_hidden_states = self.student_model(img, tokens, attention_mask,
                                                                                   a=self.a)

        ## total loss
        loss, bert_distillation_loss, bert_pred_loss, vit_loss, head_vit_loss, acc, org_acc, \
        org_bert_pred_loss = self.distillation_loss(bert_prediction_logits,
                                                    vit_hidden_states,
                                                    student_output,
                                                    student_hidden_states,
                                                    student_logits,
                                                    vit_logits,
                                                    mask,
                                                    cap)
        return loss, bert_distillation_loss, bert_pred_loss, vit_loss, head_vit_loss, acc, org_acc, org_bert_pred_loss

    def training_step(self, batch, batch_idx):
        loss, bert_distillation_loss, bert_pred_loss, vit_loss, head_vit_loss, acc, org_acc, org_bert_pred_loss = self.step(
            batch)

        if self.global_step % self.log_step == 0:
            return {"loss": loss, 'bert distillation loss': bert_distillation_loss,
                    'bert prediction loss': bert_pred_loss, 'vit loss': vit_loss, 'head vit loss': head_vit_loss,
                    'acc': acc, 'org acc': org_acc, 'org bert prediction loss': org_bert_pred_loss, 'log': True}

        return {"loss": loss, 'bert distillation loss': bert_distillation_loss, 'bert prediction loss': bert_pred_loss,
                'vit loss': vit_loss, 'head vit loss': head_vit_loss, 'acc': acc, 'org acc': org_acc,
                'org bert prediction loss': org_bert_pred_loss,
                'log': False}

    def training_step_end(self, batch_parts):
        n = self.num_gpu
        if isinstance(batch_parts, dict):
            batch_parts = [batch_parts]
            n = 1
        loss, bert_distillation_loss, bert_pred_loss, acc, org_acc, org_bert_pred_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        head_vit_loss = 0.0
        if self.vit_layers is not None:
            vit_loss = np.array([0] * len(self.vit_layers), dtype='float64')
        else:
            vit_loss = 0

        for i in range(n):
            loss += batch_parts[i]['loss']
            bert_distillation_loss += batch_parts[i]['bert distillation loss']
            bert_pred_loss += batch_parts[i]['bert prediction loss']
            vit_loss += batch_parts[i]['vit loss']
            head_vit_loss += batch_parts[i]['head vit loss']
            acc += batch_parts[i]['acc']
            org_acc += batch_parts[i]['org acc']
            org_bert_pred_loss += batch_parts[i]['org bert prediction loss']
        loss /= n
        bert_distillation_loss /= n
        bert_pred_loss /= n
        vit_loss /= n
        head_vit_loss /= n
        acc /= n
        org_bert_pred_loss /= n
        org_acc /= n

        if batch_parts[0]['log']:
            bert_log_dict = {'train/bert distillation loss ': bert_distillation_loss,
                             'train/bert prediction loss ': bert_pred_loss}

            if self.vit_layers is not None:
                vit_log_dict = {f'train/vit loss {self.vit_layers[j]}': loss for j, loss in enumerate(vit_loss)}
                total_vit_loss = vit_loss.mean()
            else:
                total_vit_loss = 0
                vit_log_dict = {}

            self.logger.experiment.log(
                {"train/epoch": self.current_epoch, "train/step loss": loss.detach().item(), **bert_log_dict,
                 **vit_log_dict, 'train/layers vit loss': total_vit_loss, 'train/head vit loss': head_vit_loss},
                step=self.global_step)
            self.logger.experiment.log({"train_metrics/acc": acc},
                                       step=self.global_step)

        return {"loss": loss, 'bert distillation loss': bert_distillation_loss,
                'bert prediction loss': bert_pred_loss, 'vit loss': vit_loss, 'head vit loss': head_vit_loss}

    def training_epoch_end(self, training_step_outputs):
        loss, bert_distillation_loss, bert_pred_loss = 0.0, 0.0, 0.0
        head_vit_loss = 0.0
        if self.vit_layers is not None:
            vit_loss = np.array([0] * len(self.vit_layers), dtype='float64')
        else:
            vit_loss = 0
        n = len(training_step_outputs)
        for out in training_step_outputs:
            loss += out['loss']
            bert_distillation_loss += out['bert distillation loss']
            bert_pred_loss += out['bert prediction loss']
            vit_loss += out['vit loss']
            head_vit_loss += out['head vit loss']
        loss /= n
        bert_distillation_loss /= n
        bert_pred_loss /= n
        vit_loss /= n
        head_vit_loss /= n

        layer_vit_loss = vit_loss.mean() if self.vit_layers is not None else 0

        print(
            "Epoch: {}, Loss: {:.5}, Bert distillation loss: {:.5}, Bert prediction loss: {:.5}, Layer Vit loss: {:.5},"
            "Head Vit loss: {:.5}".format(self.current_epoch, loss.detach().item(), bert_distillation_loss,
                                          bert_pred_loss, layer_vit_loss, head_vit_loss.detach().item()))

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss, bert_distillation_loss, bert_pred_loss, vit_loss, head_vit_loss, acc, org_acc, \
            org_bert_pred_loss = self.step(batch)
        return {"loss": loss, 'bert distillation loss': bert_distillation_loss, 'bert prediction loss': bert_pred_loss,
                'vit loss': vit_loss, 'head vit loss': head_vit_loss, 'acc': acc, 'org acc': org_acc,
                'org bert prediction loss': org_bert_pred_loss}

    def validation_step_end(self, batch_parts):
        n = self.num_gpu
        if isinstance(batch_parts, dict):
            batch_parts = [batch_parts]
            n = 1
        loss, bert_distillation_loss, bert_pred_loss, acc, org_acc, org_bert_pred_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        head_vit_loss = 0.0
        if self.vit_layers is not None:
            vit_loss = np.array([0] * len(self.vit_layers), dtype='float64')
        else:
            vit_loss = 0
        for i in range(n):
            loss += batch_parts[i]['loss']
            bert_distillation_loss += batch_parts[i]['bert distillation loss']
            bert_pred_loss += batch_parts[i]['bert prediction loss']
            vit_loss += batch_parts[i]['vit loss']
            head_vit_loss += batch_parts[i]['head vit loss']
            acc += batch_parts[i]['acc']
            org_acc += batch_parts[i]['org acc']
            org_bert_pred_loss += batch_parts[i]['org bert prediction loss']
        loss /= n
        bert_distillation_loss /= n
        bert_pred_loss /= n
        vit_loss /= n
        head_vit_loss /= n
        acc /= n
        org_bert_pred_loss /= n
        org_acc /= n

        return {"loss": loss, 'bert distillation loss': bert_distillation_loss,
                'bert prediction loss': bert_pred_loss, 'vit loss': vit_loss, 'head vit loss': head_vit_loss,
                'acc': acc, 'org acc': org_acc, 'org bert prediction loss': org_bert_pred_loss}

    def validation_epoch_end(self, validation_step_outputs):
        loss, bert_distillation_loss, bert_pred_loss, acc, org_acc, org_bert_pred_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        head_vit_loss = 0.0
        if self.vit_layers is not None:
            vit_loss = np.array([0] * len(self.vit_layers), dtype='float64')
        else:
            vit_loss = 0
        n = len(validation_step_outputs)
        for out in validation_step_outputs:
            loss += out['loss']
            bert_distillation_loss += out['bert distillation loss']
            bert_pred_loss += out['bert prediction loss']
            vit_loss += out['vit loss']
            head_vit_loss += out['head vit loss']
            acc += out['acc']
            org_acc += out['org acc']
            org_bert_pred_loss += out['org bert prediction loss']
        loss /= n
        bert_distillation_loss /= n
        bert_pred_loss /= n
        vit_loss /= n
        head_vit_loss /= n
        acc /= n
        org_bert_pred_loss /= n
        org_acc /= n

        bert_log_dict = {'val/bert distillation loss ': bert_distillation_loss,
                         'val/bert prediction loss ': bert_pred_loss}
        if self.vit_layers is not None:
            vit_log_dict = {f'val/vit loss {self.vit_layers[j]}': loss for j, loss in enumerate(vit_loss)}
            total_vit_loss = vit_loss.mean()
        else:
            total_vit_loss = 0
            vit_log_dict = {}
        # org_bert_log_dict = {"val_org/epoch": epoch, 'val_org/org acc ': org_acc,
        #                      'val_org/org bert prediction loss ': org_bert_pred_loss}

        epoch = self.current_epoch if self.global_step < 1 else self.current_epoch + 1
        self.logger.experiment.log(
            {"epoch": epoch, "val/epoch loss": loss.detach().item(), "val/ layer vit loss": total_vit_loss,
             "val/ head vit loss": head_vit_loss, **bert_log_dict, **vit_log_dict}, step=self.global_step)
        self.logger.experiment.log({"val_metrics/acc": acc},
                                   step=self.global_step)

        print(
            "VALIDATION! Epoch: {}, Loss: {:.5}, Bert distillation loss: {:.5}, Bert prediction loss: {:.5}, "
            "Layer Vit loss: {:.5}, Head Vit loss: {:.5}".format(
                self.current_epoch, loss.detach().item(), bert_distillation_loss, bert_pred_loss,
                total_vit_loss.item(), head_vit_loss.item()))

        self.save(f'exp/{self.save_dir}/{epoch}_distillation')

    def configure_optimizers(self):
        return torch.optim.Adam(self.student_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers)

    def save(self, path):
        self.student_model.vit_model.save_pretrained(path)
        # torch.save(self.student_model.state_dict(), path)


if __name__ == "__main__":
    print("START")
    # os.environ['WANDB_MODE'] = "dryrun"

    parser = argparse.ArgumentParser(description='Distillation')
    parser.add_argument('--num_gpu', type=int, default=0, help='number of gpu (default: 0)')
    parser.add_argument('--batch_size', type=int, default=45, help='batch size multiple of four')
    parser.add_argument('--vit_weights', type=str, default='../weights/vit-model_16')
    parser.add_argument('--student_weights', type=str, default='../weights/bert_vit-model_16')
    parser.add_argument('--data', type=str, default='mix', help='paired / not paired / mix (default: mix)')
    # parser.add_argument('--vit_layers', type=list, default=[], help='vit layers for loss (default: [] )')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--a', type=float, default=0)
    parser.add_argument('--vit_coeff', type=float, default=0)
    parser.add_argument('--bert_coeff', type=float, default=1)
    parser.add_argument('--p_mask', type=float, default=0.15)
    parser.add_argument('--masked', type=bool, default=True)
    parser.add_argument('--vit_head', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='models')

    arg = parser.parse_args()

    hyperparameters = {
        'num_gpu': arg.num_gpu,
        'batch_size': arg.batch_size,
        'vit_weights': arg.vit_weights,
        'student_weights': arg.student_weights,
        'data': arg.data,
        'vit_layers': [i for i in range(1, 13)],
        'weight_decay': arg.weight_decay,
        'lr': arg.lr,
        'alpha': arg.alpha,
        'a': arg.a,
        'vit_coeff': arg.vit_coeff,
        'bert_coeff': arg.bert_coeff,
        'p_mask': arg.p_mask,
        'masked': arg.masked,
        'vit_head': arg.vit_head,
        'save_dir': arg.save_dir
    }
    print(hyperparameters)

    wandb_logger = WandbLogger(name="gpu_distillation", project="pytorch-distillation", offline=False,
                               config=hyperparameters)
    train_distillation = Train_Distillation(**hyperparameters)
    wandb_logger.watch(train_distillation)
    trainer = pl.Trainer(gpus=hyperparameters['num_gpu'], max_epochs=10000, min_epochs=10000, accelerator='ddp',
                         logger=wandb_logger)
    tuner = Tuner(trainer)

    # new_batch_size = tuner.scale_batch_size(train_distillation, mode='binsearch', init_val=30)
    # train_distillation.batch_size = new_batch_size

    # with wandb.init(project="pytorch-vit", config=hyperparameters):

    trainer.fit(train_distillation)
