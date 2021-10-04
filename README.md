# Utilizing Relative Event Time to Enhance Event-Event Temporal Relation Extraction
Resources for EMNLP 2021 paper "Utilizing Relative Event Time to Enhance Event-Event Temporal Relation Extraction"

## Dependencies
- python==3.7.7
- numpy==1.18.5
- configargparse==1.2.3
- lxml==4.5.2
- pytorch==1.5
- cudatoolkit==10.2
- transformers==3.0.2
- tqdm==4.62.2

## Running script sample
**Training sample**
`python train.py --model_name roberta-large --model_type time_anchor --cache_dir ${cache_dir} --output_dir ${output_dir}  --batch_size ${batch_size} --update_batch_size ${update_batch_size} --num_train_epochs ${num_train_epochs} --lr ${lr} --seed ${seed} --dataset matres --do_train`

**Evaluate sample**
`python train.py --model_name roberta-large --model_type time_anchor --cache_dir ${cache_dir} --output_dir ${output_dir}  --batch_size ${batch_size} --update_batch_size ${update_batch_size} --num_train_epochs ${num_train_epochs} --lr ${lr} --seed ${seed} --dataset matres --do_eval`

## Reference
```
@inproceedings{emnlp-2021-temprel,
	title = "Utilizing Relative Event Time to Enhance Event-Event Temporal Relation Extraction",
	author = "Wen, Haoyang and Ji, Heng",
	booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
	year = 2021,
}
```

## Acknowledgement
Some part of the code and the pre-processed data from the repository of the paper ["An Improved Neural Baseline for Temporal Relation Extraction."](https://github.com/qiangning/NeuralTemporalRelation-EMNLP19) Qiang Ning, Sanjay Subramanian, and Dan Roth.
