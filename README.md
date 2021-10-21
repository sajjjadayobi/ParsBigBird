<span align="center">
    <a href="https://huggingface.co/SajjadAyoubi/"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=SajjadAyoubi&color=yellow"></a>
    <a href="https://colab.research.google.com/github/sajjjadayobi/PersianQA/blob/main/notebooks/Demo.ipynb"><img src="https://img.shields.io/static/v1?label=Colab&message=Fine-tuning Example&logo=Google%20Colab&color=f9ab00"></a>
</span>


# ParsBigBird:🐦 Persian Bert for long sequences
[Bert](https://arxiv.org/abs/1810.04805) and [ParsBert](https://arxiv.org/abs/2005.12515) can handle texts of token lengths up to 512, but many tasks such as summarization and question answering require longer texts. [BigBird](https://arxiv.org/abs/2007.14062) model can handle text until **4096** due to sparse attention, in this work we've trained big bird model for Persian language to process texts up to 4096 in Persian(farsi) language

## Evaluation: 🌡️
we have evaluated the model on two tasks with different seqence lengths, SnappFood Sentiment Analysis dataset and PerisanQA Question-Answering dataset

|         Name       | Params |    SnappFood (Acc)   |  PersianQA (F1)   |
| :----------------: | :----: | :------------------: | :---------------: |
| [distil-bigbird-fa]() |  100M  | -                    |        -          |
| [bert-base-fa]()   |  162M  |        87.98%        |       70.06%      |

- we have evaluated our model on two dataset, one with long texts and another with small texts to show its ability to handle both of them
- the model performs compatible with ParsBert while being 2x smaller 


## How to use❓

### As Contexulized Word Embedding 
```python
from transformers import BigBirdModel, TFBigBirdModel, AutoTokenizer
MODEL_NAME = "SajjadAyoubi/bigbird-fa-base"

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
MODEL_NAME = 'SajjadAyoubi/bigbird-fa-base'

fill = pipeline('fill-mask', model=MODEL_NAME, tokenizer=MODEL_NAME)
results = fill('تهران پایتخت [MASK] است.')
print(results[0]['token_str'])
>>> 'ایران'
```


## Pretraining details: 🔭
It is a pretrained model on Persian section of Oscar dataset using a masked language modeling (MLM) objective. Following the original BERT training, 15% of tokens were masked. It was introduced in this [paper]() and first released in this [repository](). Document longer than 4096 were split into multiple documents and documents that were much smaller than 4096 were joined using the [SEP] token. Model is warm started from Distil-BERT’s checkpoint. It doesn't matter how many tokens is input text in block_sparse mode it just attends to 256 tokens. 
for more details you can take a look at config.json at model card in 

## Fine Tuning recommendations: 🐤
this model needs a reasonable amount of GPU memory so in order to have a reasonable batch size, `gradient_checkpointing` and `gradient_accumulation_steps` are recommended. as far as this model isn't really big it's a good idea to first fine tune it on your dataset using Masked LM objective (also called intermediate fine tuning) lastly fine tuned on our main task. Also it’s recommended to use original_full (instead of block sparse) till 512 seqlen.

### Fine tuning example 👷‍♂️👷‍♀️
DigiKala Text Classification on Colab

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
