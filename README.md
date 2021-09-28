# ParsBigBird: a Persian transformer for long sequences
the original **Bert** and [ParsBert]() can handle texts with tokens length until 512 but in many tasks like summarization or question answering we need to have texts with bigger input. [BigBird]() model can handle text until **4096** due to sparse attention, in this work we've trained big bird model for Persian langiage

<p align="center">
  <img src="https://s4.uupload.ir/files/bird_88cg.png">
</p>


## How to use?

```python
from transformers import AutoTokenizer, AutoModel, TFAutoModel

MODEL_NAME = "SajjadAyoubi/pars-big-bird-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tf_model = TFAutoModel.from_pretrained(MODEL_NAME)
pt_model = AutoModel.from_pretrained(MODEL_NAME)
```

## Pretraining details



## Contact us
If you have a technical question regarding the model, pretraining, code or publication, please create an issue in the repository. This is the fastest way to reach us.

## Citation
we didn't publish any papers on the work. However, if you did, please cite us properly with an entry like one below.
```bibtex
@misc{ParsBigBird,
  author          = {Ayoubi, Sajjad},
  title           = {ParsBigBird: a Persian transformer for long sequences},
  year            = 2021,
  publisher       = {GitHub},
  journal         = {GitHub repository},
  howpublished    = {\url{https://github.com/SajjjadAyobi/PersianQA}},
}
```
