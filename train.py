import os
import json
import torch
import logging
import configargparse
import torch
import numpy as np
import subprocess

from math import ceil
from tqdm import tqdm
from typing import List, Optional
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.dataset import random_split

from utils import setup_tokenizer, set_random_seed, setup_cuda_device,\
    setup_scheduler_optimizer, get_label_type
from TemporalDataSet import temprel_set
from model import prepare_model


def _get_validated_args(input_args: Optional[List[str]] = None):
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add_argument("--dataset", type=str, default="matres",
                        choices=["matres"], required=True,
                        help="Choose a dataset")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Choose a pretrained base model.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Choose a relation classification model.")
    parser.add_argument("--model_weights", type=str, default=None,
                        help="The trained model weights.")
    parser.add_argument("--cache_dir", type=str,
                        help="Cache directory for pretrained models.")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for parameters from trained model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Not to use CUDA when available.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for each running.")
    parser.add_argument("--update_batch_size", type=int, default=32,
                        help="Batch size for each model update.")
    parser.add_argument("--lr", type=float, default=4e-5,
                        help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="The number of epochs for pretrained model.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="The maximum gradient norm.")

    parser.add_argument("--do_train", action='store_true',
                        help="Perform training")
    parser.add_argument("--do_eval", action='store_true',
                        help="Perform evaluation on test set")

    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta 1 parameters (b1, b2) for optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta 1 parameters (b1, b2) for optimizer.")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Epsilon for numerical stability for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Decoupled weight decay to apply on optimizer.")
    parser.add_argument("--num_warmup_ratio", type=float, default=0.1,
                        help="The number of steps for the warmup phase")

    args = parser.parse_args(input_args)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    
    return args


def _get_tensorset(tokenizer):
    logging.info("***** Loading Dataset *****\n")
    traindevset = temprel_set("data/trainset-temprel.xml")
    traindev_tensorset = traindevset.to_tensor(tokenizer=tokenizer)
    train_idx = list(range(len(traindev_tensorset)-1852))
    dev_idx = list(range(len(traindev_tensorset)-1852, len(traindev_tensorset)))
    train_tensorset = Subset(traindev_tensorset, train_idx)
    dev_tensorset = Subset(traindev_tensorset, dev_idx) #Last 21 docs
    logging.info(f"All = {len(traindev_tensorset)}, Train={len(train_tensorset)}, Dev={len(dev_tensorset)}")

    testset = temprel_set("data/testset-temprel.xml")
    test_tensorset = testset.to_tensor(tokenizer=tokenizer)
    logging.info(f"Test = {len(test_tensorset)}")
    return train_tensorset, dev_tensorset, test_tensorset


def _gather_model_inputs(model_type, batch):
    inputs = {'input_ids': batch[0],
              'attention_mask': batch[1],
              'event_ix': batch[2],
              'labels': batch[3]}
    if model_type == 'time_anchor':
        return inputs
    else:
        raise ValueError("Invalid model type")


def calc_f1(predicted_labels, all_labels, label_type):
    confusion = np.zeros((len(label_type), len(label_type)))
    for i in range(len(predicted_labels)):
        confusion[all_labels[i]][predicted_labels[i]] += 1

    acc = 1.0 * np.sum([confusion[i][i] for i in range(4)]) / np.sum(confusion)
    true_positive = 0
    for i in range(len(label_type)-1):
        true_positive += confusion[i][i]
    prec = true_positive/(np.sum(confusion)-np.sum(confusion,axis=0)[-1])
    rec = true_positive/(np.sum(confusion)-np.sum(confusion[-1][:]))
    f1 = 2*prec*rec / (rec+prec)

    return acc, prec, rec, f1, confusion


def evaluate(model, model_type, dataloader, device, dataset, threshold=None):
    model.eval()
    label_type = get_label_type(dataset)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch = [x.to(device) for x in batch]
            inputs = _gather_model_inputs(model_type, batch)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]
            all_logits.append(logits)
            all_labels.append(inputs['labels'])

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    predicted_logits, predicted_labels = torch.max(all_logits, dim=1)

    acc, prec, rec, f1, confusion = calc_f1(predicted_labels, all_labels, label_type)
    return acc, prec, rec, f1, confusion, threshold


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, device, n_gpu):
    num_training_steps_per_epoch = ceil(len(train_dataloader.dataset)/float(args.update_batch_size))
    num_training_steps = args.num_train_epochs * num_training_steps_per_epoch

    scheduler, optimizer = setup_scheduler_optimizer(model=model,
                                                     num_warmup_ratio=args.num_warmup_ratio,
                                                     num_training_steps=num_training_steps,
                                                     lr=args.lr, beta1=args.beta1,
                                                     beta2=args.beta2, eps=args.eps,
                                                     weight_decay=args.weight_decay)

    global_step = 0
    best_acc = 0.
    update_per_batch = args.update_batch_size // args.batch_size
    for epoch in range(1, args.num_train_epochs+1, 1):
        model.train()
        global_loss = 0.
        for i, batch in tqdm(enumerate(train_dataloader),
                          desc=f'Running train for epoch {epoch}',
                          total=len(train_dataloader)):
            batch = [x.to(device) for x in batch]
            inputs = _gather_model_inputs(args.model_type, batch)
            outputs = model(**inputs)
            loss, logits = outputs[0], outputs[1]

            loss /= update_per_batch
            loss.backward()
            if (i+1) % update_per_batch == 0 or (i+1) == len(train_dataloader):
                # global_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                global_loss = 0
        logging.info(f"Evaluation for epoch {epoch}")
        dev_metrics = evaluate(model, args.model_type, dev_dataloader, device, args.dataset)
        dev_acc, dev_prec, dev_rec, dev_f1, dev_confusion, dev_threshold = dev_metrics
        logging.info(f"Acc={dev_acc}, Precision={dev_prec}, Recall={dev_rec}, F1={dev_f1}")
        logging.info(f"Confusion={dev_confusion}")
        if dev_f1 > best_acc:
            logging.info(f"New best, dev_f1={dev_f1} > best_f1={best_acc}")
            best_acc = dev_f1
            if n_gpu > 1:
                model.module.save_pretrained(args.output_dir)
            else:
                model.save_pretrained(args.output_dir)
            logging.info(f"Best model saved in {args.output_dir}")


def main(input_args: Optional[List[str]] = None):
    args = _get_validated_args(input_args)
    try:
        logging.info(f"Current Git Hash: {get_git_revision_hash()}")
    except:
        pass
    device, n_gpu = setup_cuda_device(args.no_cuda)
    set_random_seed(args.seed, n_gpu)

    model_name = args.model_name if args.cache_dir is None else \
                 os.path.join(args.cache_dir, args.model_name)
    model = prepare_model(model_name, args.model_type, device, args.model_weights, args.dataset)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    tokenizer = setup_tokenizer(model_name=args.model_name, cache_dir=args.cache_dir)
    if args.dataset == "matres":
        train_tensorset, dev_tensorset, test_tensorset = _get_tensorset(tokenizer)

    train_dataloader = DataLoader(train_tensorset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_tensorset, batch_size=args.batch_size//2, shuffle=False)
    test_dataloader = DataLoader(test_tensorset, batch_size=args.batch_size//2, shuffle=False)

    if args.do_train:
        train(args, model, train_dataloader, dev_dataloader, test_dataloader, device, n_gpu)
    
    if args.do_eval:
        test_metrics = evaluate(model, args.model_type, test_dataloader, device, args.dataset)
        test_acc, test_prec, test_rec, test_f1, test_confusion, test_threshold = test_metrics
        logging.info(f"Acc={test_acc}, Precision={test_prec}, Recall={test_rec}, F1={test_f1}")
        logging.info(f"Confusion={test_confusion}")


if __name__ == "__main__":
    main()