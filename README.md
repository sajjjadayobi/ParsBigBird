# ParsBigBird:🐦 a Persian transformer for long sequences
**Bert** and [ParsBert]() can handle texts of token lengths up to 512, but many tasks such as summarization and question answering require longer texts. [BigBird]() model can handle text until **4096** due to sparse attention, in this work we've trained big bird model for Persian language


## How to use❓

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

## Pretraining details: 🔭
It is a pretrained model on Persian section of Oscar dataset using a masked language modeling (MLM) objective. Following the original BERT training, 15% of tokens were masked. It was introduced in this [paper]() and first released in this [repository](). Model is warm started from Distil-BERT’s checkpoint. It doesn't matter how many tokens is input text in block_sparse mode it just attends to 256 tokens. Also it’s recommended to use original_full (instead of block sparse) till 512 seqlen.

## Fine Tuning recommendations: 🐤
this model needs a reasonable amount of GPU memory so in order to have a reasonable batch size, `gradient_checkpointing` and `gradient_accumulation_steps` are recommended. 

## Contact us: 🤝
If you have a technical question regarding the model, pretraining, code or publication, please create an issue in the repository. This is the fastest way to reach us.

## Citation: ↩️
we didn't publish any papers on the work. However, if you did, please cite us properly with an entry like one below.
```bibtex
@misc{ParsBigBird,
  author          = {Ayoubi, Sajjad},
  title           = {ParsBigBird: a Persian transformer for long sequences},
  year            = 2021,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/ParsBigBird}},
}
```
