<span align="center">
    <a href="https://huggingface.co/SajjadAyoubi/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=SajjadAyoubi&color=yellow"></a>
    <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Fine-tuning Example&logo=Google%20Colab&color=f9ab00"></a>
</span>


# ParsBigBird: Persian Bert For **Long-Range** Sequences
The [Bert](https://arxiv.org/abs/1810.04805) and [ParsBert](https://arxiv.org/abs/2005.12515) algorithms can handle texts with token lengths of up to 512, however, many tasks such as summarizing and answering questions require longer texts. In our work, we have trained the [BigBird](https://arxiv.org/abs/2007.14062) model for the Persian language to process texts up to 4096 in the Farsi (Persian) language using sparse attention.

## Evaluation: 🌡️
We have evaluated the model on two tasks with different seqence lengths, SnappFood Sentiment Analysis dataset and PerisanQA Question-Answering dataset

|         Name       | Params |    SnappFood (Acc)   |  PersianQA (F1)   |
| :----------------: | :----: | :------------------: | :---------------: |
| [distil-bigbird-fa]() |  100M  | -                    |        -          |
| [bert-base-fa]()   |  162M  |        87.98%        |       70.06%      |

- we have evaluated our model on two dataset, one with long texts and another with small texts to show its ability to handle both of them
- the model performs compatible with ParsBert while being 2x smaller 


## How to use❓

### As Contexulized Word Embedding 
```python
from transformers import BigBirdModel, AutoTokenizer

MODEL_NAME = "SajjadAyoubi/bigbird-fa-uncased"
# by default its in `block_sparse` block_size=32
model = BigBirdModel.from_pretrained(MODEL_NAME, block_size=32)
# you can use full attention like the following: use this when input isn't longer than 512
model = BigBirdModel.from_pretrained(MODEL_NAME, attention_type="original_full")

text = "😃 امیدوارم مدل بدردبخوری باشه چون خیلی طول کشید تا ترین بشه"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokens = tokenizer(text, return_tensors='pt')
output = model(**tokens) # contextualized embedding
```

### As Fill Blank
```python
from transformers import pipeline

MODEL_NAME = 'SajjadAyoubi/bigbird-fa-uncased'
fill = pipeline('fill-mask', model=MODEL_NAME, tokenizer=MODEL_NAME)
results = fill('تهران پایتخت [MASK] است.')
print(results[0]['token_str'])
>>> 'ایران'
```

## Pretraining details: 🔭
This model was pretrained using a masked language model (MLM) objective on the Persian section of the Oscar dataset. Following the original BERT training, 15% of tokens were masked. This was first described in this [paper](https://arxiv.org/abs/2007.14062) and released in this [repository](https://github.com/google-research/bigbird). Documents longer than 4096 were split into multiple documents, while documents much smaller than 4096 were merged using the [SEP] token. Model is warm started from `distilbert-fa`’s [checkpoint](https://huggingface.co/HooshvareLab/distilbert-fa-zwnj-base). 
- For more details, you can take a look at config.json at the model card in 🤗 Model Hub

## Fine Tuning Recommendations: 🐤
Due to the model's memory requirements, `gradient_checkpointing` and `gradient_accumulation` should be used to maintain a reasonable batch size. Considering this model isn't really big, it's a good idea to first fine-tune it on your dataset using Masked LM objective (also called intermediate fine-tuning) before implementing the main task. In block_sparse mode, it doesn't matter how many tokens are input. It just attends to 256 tokens. Furthermore, original_full should be used up to 512 sequence lengths (instead of block sparse).

### Fine Tuning Examples 👷‍♂️👷‍♀️

| Dataset                               | Fine Tuning Example                                          |
| ------------------------------------- | ------------------------------------------------------------ |
| Digikala Magazine Text Classification | <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Fine-tuning Example&logo=Google%20Colab&color=f9ab00"></a> |


## Contact us: 🤝
If you have a technical question regarding the model, pretraining, code or publication, please create an issue in the repository. This is the fastest way to reach us.

## Citation: ↩️
we didn't publish any papers on the work. However, if you did, please cite us properly with an entry like one below.
```bibtex
@misc{ParsBigBird,
  author          = {Ayoubi, Sajjad},
  title           = {ParsBigBird: Persian Bert for long sequences},
  year            = 2021,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/ParsBigBird}},
}
```
