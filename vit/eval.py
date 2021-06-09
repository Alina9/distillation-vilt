import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DataLoader import CocoCaptions, DatasetWrapper
from train1 import Train_Vit
from tqdm import tqdm
import wandb
import argparse
import csv


def recall_at_k(indices, targets):
    ## for image retrieval
    if len(targets.shape) == 1:
        return (indices == targets).sum(0).float().mean()
    ## for caption retrieval
    num_elements, num_targets = targets.shape
    targets = targets.transpose(0, 1).reshape(-1, 1)
    indices = indices.repeat(num_targets, 1)
    overlap = targets == indices
    recalls = sum(overlap.split(num_elements, 0), 0).sum(1) > 0
    recall = recalls.float().mean()
    return recall


def get_intermediate_result(score_matrix, images2caption):
    images2caption = torch.cat([torch.tensor(images2caption[i]).unsqueeze(0) for i in images2caption.keys()], 0).to(
        device)
    recalls_image_cap = {k: [] for k in K}
    for k in K:
        _, topk_caption = torch.topk(score_matrix, k, 1)
        recalls_image_cap[k] = recall_at_k(topk_caption, images2caption)
    # print(f"Обработано {score_matrix.shape[0]} картинок")
    # print("Для картинки ищем текст:")
    # print(recalls_image_cap)
    # print(recalls_image_cap[1])
    wandb.log({"Обработано картинок": score_matrix.shape[0]}, step=score_matrix.shape[0])
    wandb.log({"k=1": recalls_image_cap[1].item()}, step=score_matrix.shape[0])
    wandb.log({"k=5": recalls_image_cap[5].item()}, step=score_matrix.shape[0])
    wandb.log({"k=10": recalls_image_cap[10].item()}, step=score_matrix.shape[0])


def get_recalls(model, eval_dataset, K, SIZE):
    recalls_image_cap = {k: [] for k in K}
    recalls_cap_image = {k: [] for k in K}
    score_matrix, images2caption, caption2image = get_scores_matrix(model, eval_dataset, SIZE)

    for k in K:
        _, topk_caption = torch.topk(score_matrix, k, 1)
        _, topk_images = torch.topk(score_matrix, k, 0)
        recalls_image_cap[k] = recall_at_k(topk_caption, images2caption)
        recalls_cap_image[k] = recall_at_k(topk_images, caption2image)

    wandb.log({"Обработано картинок": score_matrix.shape[0]}, step=score_matrix.shape[0])
    wandb.log({"k=1": recalls_image_cap[1].item()}, step=score_matrix.shape[0])
    wandb.log({"k=5": recalls_image_cap[5].item()}, step=score_matrix.shape[0])
    wandb.log({"k=10": recalls_image_cap[10].item()}, step=score_matrix.shape[0])
    return recalls_image_cap, recalls_cap_image


def get_scores_matrix(model, eval_dataset, SIZE):
    score_matrix = torch.tensor([]).to(device)
    images2caption = {}
    caption2image = torch.empty(5 * SIZE).to(device)
    image_id2id = {}
    id = -1  ## порядковый номер картинки

    for i in tqdm(range(5 * SIZE)):  ## порядковый номер текста
        image, img_id = eval_dataset.get_image(i)
        if img_id in image_id2id:
            caption2image[i] = image_id2id[img_id]
            images2caption[image_id2id[img_id]].append(i)
            continue
        if score_matrix.shape[0] > 0 and score_matrix.shape[0] % 50 == 0:
            get_intermediate_result(score_matrix, images2caption)
        id += 1
        image_id2id[img_id] = id
        images2caption[id] = []
        images2caption[id].append(i)
        caption2image[i] = id

        scores_for_one_image = torch.tensor([]).to(device)
        for _, cap in loader:
            batch_size = len(cap)
            img_batch = torch.cat([image.unsqueeze(0)] * batch_size, 0)
            batch = (img_batch.to(device), cap)
            with torch.no_grad():
                y_score, y_pred = model(batch)
            scores_for_one_image = torch.cat((scores_for_one_image, y_score), 0)
        score_matrix = torch.cat((score_matrix, scores_for_one_image.unsqueeze(0)), 0)

    images2caption = torch.cat([torch.tensor(images2caption[i]).unsqueeze(0) for i in images2caption.keys()], 0).to(
        device)

    return score_matrix, images2caption, caption2image.long()


if __name__ == "__main__":
    os.environ['WANDB_MODE'] = "dryrun"
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--model_weights', type=str, default="../weights/78_vit.pkl")
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--device', type=str, default="cuda:0")

    arg = parser.parse_args()

    device = torch.device(arg.device if torch.cuda.is_available() else "cpu")
    ## number of images
    SIZE = arg.size

    eval_dataset = CocoCaptions(root="../../val2014",
                                annFile='../../annotations/captions_val2014.json',
                                transform=transform,
                                start=-SIZE * 5)

    loader = DataLoader(eval_dataset, batch_size=arg.batch_size, shuffle=False, num_workers=48)

    train_vit = Train_Vit(num_gpu=1, model_weight='../weights/bert_vit-model_16', p_mask=0, img_mask=False)
    train_vit.to(device)

    train_vit.load_weights(arg.model_weights)

    K = [1, 5, 10]
    with wandb.init(project="eval", name=f"{arg.model_weights}_vit", config={"weights": arg.model_weights}):
        recalls_image_cap, recalls_cap_image = get_recalls(train_vit, eval_dataset, K, SIZE)
    print(f"Номер эпохи: {w}")
    print("Для картинки ищем текст:")
    print(recalls_image_cap)
    print()
    print("Для текста ищем картинку")
    print(recalls_cap_image)
