import logging
import random
import torch
import numpy as np

from os import path
from enum import Enum
from math import ceil
from typing import Optional

from transformers import RobertaTokenizerFast, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


def get_label_type(dataset):
    if dataset == 'matres':
        label_type = LabelType
    else:
        raise ValueError(f"Dataset {dataset} not support")
    return label_type


class LabelType(Enum):
    BEFORE = 0
    AFTER = 1
    EQUAL = 2
    VAGUE = 3

    @staticmethod
    def to_class_index(label_type):
        for label in LabelType:
            if label_type == label.name:
                return label.value


def setup_cuda_device(no_cuda):
    if no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        n_gpu = 0
    else:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    return device, n_gpu


def set_random_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Set random seed to {seed}")


def setup_tokenizer(model_name: str, cache_dir: Optional[str] = None):
    logging.info("***** Setting up tokenizer *****\n")

    model_path = model_name
    if cache_dir:
        logging.info("Loading tokenizer from cache files")
        model_path = path.join(cache_dir, model_name)
        if not path.isdir(model_path):
            raise FileNotFoundError(f"No cache directory {model_path} exists.")
    else:
        logging.info(f"Loading {model_path} tokenizer")
        print(cache_dir)
    do_lower_case = 'uncased' in model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, do_lower_case=do_lower_case)

    return tokenizer
    
def setup_scheduler_optimizer(model: torch.nn.Module, num_warmup_ratio: float, num_training_steps: int,
                              lr: Optional[float] = 2e-5, beta1: Optional[float] = 0.9,
                              beta2: Optional[float] = 0.999, eps: Optional[float] = 1e-8,
                              weight_decay: Optional[float] = 0):
    logging.info("***** Setting up scheduler and optimizer *****\n")

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters_to_optimize if
                    not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps, betas=(beta1, beta2))

    num_warmup_steps = ceil(num_warmup_ratio * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=num_warmup_steps, 
                                                num_training_steps=num_training_steps)
    
    logging.info(f"Initialized optimizer {optimizer}")
    logging.info(f"Initialized scheduler {scheduler}")

    return scheduler, optimizer


TRAIN_DOC_LIST = ['ABC19980120.1830.0957',
 'wsj_0791',
 'VOA19980305.1800.2603',
 'VOA19980303.1600.0917',
 'wsj_0806',
 'wsj_0709',
 'wsj_0938',
 'wsj_0340',
 'wsj_0158',
 'wsj_1039',
 'AP900815-0044',
 'wsj_0927',
 'wsj_0332',
 'NYT19980206.0466',
 'wsj_0542',
 'wsj_0136',
 'wsj_0695',
 'VOA19980501.1800.0355',
 'wsj_0551',
 'ABC19980114.1830.0611',
 'wsj_0650',
 'PRI19980121.2000.2591',
 'wsj_0660',
 'APW19980501.0480',
 'wsj_0667',
 'wsj_0073',
 'VOA19980303.1600.2745',
 'wsj_0811',
 'wsj_0798',
 'APW19980322.0749',
 'wsj_0189',
 'wsj_0583',
 'wsj_0187',
 'CNN19980126.1600.1104',
 'wsj_0923',
 'wsj_0346',
 'wsj_0906',
 'wsj_0165',
 'APW19980308.0201',
 'wsj_0745',
 'wsj_0534',
 'wsj_0169',
 'wsj_0122',
 'wsj_0167',
 'wsj_0144',
 'wsj_0760',
 'NYT19980402.0453',
 'wsj_0471',
 'PRI19980205.2000.1890',
 'ea980120.1830.0456',
 'wsj_0175',
 'wsj_0751',
 'AP900816-0139',
 'wsj_0768',
 'wsj_0151',
 'NYT19980206.0460',
 'wsj_0124',
 'wsj_0713',
 'wsj_1003',
 'wsj_0786',
 'wsj_0778',
 'wsj_0736',
 'wsj_0325',
 'wsj_0348',
 'wsj_0176',
 'APW19980213.1380',
 'wsj_1011',
 'wsj_0329',
 'wsj_0610',
 'wsj_1042',
 'wsj_0263',
 'wsj_0752',
 'wsj_0292',
 'CNN19980223.1130.0960',
 'wsj_0575',
 'APW19980227.0468',
 'wsj_0679',
 'APW19980626.0364',
 'wsj_0068',
 'wsj_0674',
 'wsj_0157',
 'NYT19980212.0019',
 'PRI19980115.2000.0186',
 'wsj_0184',
 'wsj_0585',
 'wsj_0555',
 'ABC19980108.1830.0711',
 'wsj_0981',
 'wsj_0127',
 'wsj_0816',
 'wsj_0533',
 'wsj_0685',
 'wsj_0904',
 'ea980120.1830.0071',
 'wsj_1031',
 'APW19980301.0720',
 'wsj_0950',
 'wsj_0781',
 'wsj_0612',
 'wsj_0266',
 'APW19980227.0476',
 'CNN19980213.2130.0155',
 'wsj_0324',
 'wsj_0661',
 'VOA19980331.1700.1533',
 'wsj_0376',
 'wsj_1013',
 'PRI19980306.2000.1675',
 'wsj_0924',
 'wsj_0027',
 'wsj_1040',
 'wsj_0168',
 'wsj_0570',
 'wsj_0316',
 'wsj_0150',
 'wsj_0675',
 'wsj_0527',
 'wsj_1006',
 'APW19980227.0487',
 'wsj_0991',
 'wsj_0805',
 'wsj_0152',
 'APW19980213.1320',
 'CNN19980227.2130.0067',
 'wsj_0907',
 'wsj_0815',
 'wsj_0670',
 'wsj_1038',
 'wsj_0541',
 'wsj_0557',
 'PRI19980216.2000.0170',
 'wsj_0172',
 'APW19980418.0210',
 'wsj_0810',
 'wsj_0321',
 'wsj_0135',
 'wsj_0313',
 'SJMN91-06338157',
 'wsj_1025',
 'wsj_0106',
 'APW19980227.0494',
 'wsj_1073',
 'wsj_0928',
 'wsj_0706',
 'APW19980227.0489',
 'wsj_0356',
 'wsj_0637',
 'WSJ910225-0066',
 'wsj_0032',
 'wsj_0918',
 'wsj_0584',
 'wsj_1008',
 'PRI19980213.2000.0313',
 'NYT19980424.0421',
 'wsj_0173',
 'wsj_0520',
 'ABC19980304.1830.1636',
 'wsj_1014',
 'wsj_0026',
 'wsj_0662',
 'wsj_0568',
 'wsj_1035',
 'wsj_0160',
 'wsj_0505',
 'wsj_0558',
 'wsj_1033',
 'wsj_0006',
 'APW19980219.0476',
 'ed980111.1130.0089',
 'wsj_0161',
 'wsj_0762',
 'CNN19980222.1130.0084',
 'APW19980306.1001',
 'WSJ900813-0157',
 'wsj_0171',
 'wsj_0344',
 'wsj_0973',
 'wsj_0159',
 'APW19980213.1310',
 'wsj_0132',
 'PRI19980303.2000.2550',
 'wsj_0586',
 'NYT20000406.0002',
 'NYT19990419.0515',
 'XIE19990210.0079',
 'XIE19980809.0010',
 'NYT19981026.0446',
 'XIE19990313.0173',
 'APW19981205.0374',
 'XIE19980812.0062',
 'APW19980809.0700',
 'APW19990410.0123',
 'APW19991008.0151',
 'APW19980811.0474',
 'NYT20000113.0267',
 'APW19991008.0265',
 'APW19980826.0389',
 'APW19980813.1117',
 'XIE19990227.0171',
 'APW19990607.0041',
 'XIE19980808.0188',
 'NYT19981120.0362',
 'APW20000405.0276',
 'NYT19980907.0112',
 'NYT19990505.0443',
 'APW20000403.0057',
 'NYT20000403.0463',
 'XIE19980808.0049',
 'XIE19990313.0229',
 'NYT20000424.0319',
 'APW199980817.1193',
 'XIE19980808.0031',
 'NYT19981121.0173',
 'NYT19981025.0216',
 'NYT19990312.0271',
 'APW19980810.0907',
 'APW19990206.0090',
 'APW20000210.0328',
 'XIE19990313.0031',
 'NYT20000105.0325',
 'XIE19980821.0077',
 'XIE19980814.0294',
 'APW20000106.0064',
 'APW19980930.0425',
 'APW19990506.0155',
 'APW20000107.0318',
 'APW19991024.0075',
 'NYT19981025.0188',
 'APW20000128.0316',
 'APW20000107.0088',
 'APW19990216.0198',
 'APW20000124.0182',
 'NYT20000224.0173',
 'APW19980818.0515']


DEV_DOC_LIST = ['APW20000115.0031',
 'APW20000401.0150',
 'NYT20000329.0359',
 'APW19990122.0193',
 'NYT20000106.0007',
 'APW19980820.1428',
 'NYT20000601.0442',
 'NYT20000330.0406',
 'APW20000115.0209',
 'NYT20000414.0296',
 'APW19980807.0261',
 'APW19990507.0207',
 'XIE19981203.0008',
 'APW19990312.0251',
 'APW19980808.0022',
 'APW20000417.0031',
 'APW20000328.0257',
 'APW20000216.0193',
 'APW20000216.0272',
 'XIE19980808.0060',
 'APW19980911.0475']