# dataset-generator
---
license: mit
datasets:
- squad_v2
- quac
language:
- en
widget:
- text: >-
    when: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes
    ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an
    Argentine professional footballer who plays as a forward for and captains
    both Major League Soccer club Inter Miami and the Argentina national team.
    Widely regarded as one of the greatest players of all time, Messi has won a
    record seven Ballon d'Or awards[note 2] and a record six European Golden
    Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving
    the club in 2021, he had spent his entire professional career with
    Barcelona, where he won a club-record 34
- text: >-
    where: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes
    ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an
    Argentine professional footballer who plays as a forward for and captains
    both Major League Soccer club Inter Miami and the Argentina national team.
    Widely regarded as one of the greatest players of all time, Messi has won a
    record seven Ballon d'Or awards[note 2] and a record six European Golden
    Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving
    the club in 2021, he had spent his entire professional career with
    Barcelona, where he won a club-record 34
- text: >-
    how: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes
    ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an
    Argentine professional footballer who plays as a forward for and captains
    both Major League Soccer club Inter Miami and the Argentina national team.
    Widely regarded as one of the greatest players of all time, Messi has won a
    record seven Ballon d'Or awards[note 2] and a record six European Golden
    Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving
    the club in 2021, he had spent his entire professional career with
    Barcelona, where he won a club-record 34
- text: >-
    what: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes
    ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an
    Argentine professional footballer who plays as a forward for and captains
    both Major League Soccer club Inter Miami and the Argentina national team.
    Widely regarded as one of the greatest players of all time, Messi has won a
    record seven Ballon d'Or awards[note 2] and a record six European Golden
    Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving
    the club in 2021, he had spent his entire professional career with
    Barcelona, where he won a club-record 34
- text: >-
    where: Egypt (Egyptian Arabic: مصر Maṣr Egyptian Arabic pronunciation:
    [mɑsˤr]), officially the Arab Republic of Egypt, is a transcontinental
    country spanning the northeast corner of Africa and the Sinai Peninsula in
    the southwest corner of Asia. It is bordered by the Mediterranean Sea to the
    north, the Gaza Strip of Palestine and Israel to the northeast, the Red Sea
    to the east, Sudan to the south, and Libya to the west. The Gulf of Aqaba in
    the northeast separates Egypt from Jordan and Saudi Arabia. Cairo is the
    capital and largest city of Egypt, while Alexandria, the second-largest
    city, is an important industrial and tourist hub at the Mediterranean
    coast.[11] At approximately 100 million inhabitants, Egypt is the 14th-most
    populated country in the world, and the third-most populated in Africa,
    behind Nigeria and Ethiopia.
- text: >-
    where: There is evidence of rock carvings along the Nile terraces and in
    desert oases. In the 10th millennium BCE, a culture of hunter-gatherers and
    fishers was replaced by a grain-grinding culture. Climate changes or
    overgrazing around 8000 BCE began to desiccate the pastoral lands of Egypt,
    forming the Sahara. Early tribal peoples migrated to the Nile River where
    they developed a settled agricultural economy and more centralized society.
- text: >-
    when: By about 6000 BCE, a Neolithic culture took root in the Nile
    Valley.[31] During the Neolithic era, several predynastic cultures developed
    independently in Upper and Lower Egypt. The Badarian culture and the
    successor Naqada series are generally regarded as precursors to dynastic
    Egypt. The earliest known Lower Egyptian site, Merimda, predates the
    Badarian by about seven hundred years. Contemporaneous Lower Egyptian
    communities coexisted with their southern counterparts for more than two
    thousand years. The earliest known evidence of Egyptian hieroglyphic
    inscriptions appeared during the predynastic period on Naqada III pottery
    vessels, dated to about 3200 BCE.[32]
- text: >-
    whose : or the next three millennia. Egyptian culture flourished during this
    long period and remained distinctively Egyptian in its religion, arts,
    language and customs. The first two ruling dynasties of a unified Egypt set
    the stage for the Old Kingdom period, c. 2700–2200 BCE, which constructed
    many pyramids, most notably the Third Dynasty pyramid of Djoser and the
    Fourth Dynasty Giza pyramids.
- text: >-
    who:The First Intermediate Period ushered in a time of political upheaval
    for about 150 years.[33] Stronger Nile floods and stabilisation of
    government, however, brought back renewed prosperity for the country in the
    Middle Kingdom c. 2040 BCE, reaching a peak during the reign of Pharaoh
    Amenemhat III. A second period of disunity heralded the arrival of the first
    foreign ruling dynasty in Egypt, that of the Semitic Hyksos. The Hyksos
    invaders took over much of Lower Egypt around 1650 BCE and founded a new
    capital at Avaris. They were driven out by an Upper Egyptian force led by
    Ahmose I, who founded the Eighteenth Dynasty and relocated the capital from
    Memphis to Thebes.
library_name: transformers
tags:
- generate answers
- question generator
- generate text
- nlp
- dataset maker
- flan t5
- t5
---

# Model Card for QA_GeneraToR

# my fine tuned model
>This model is fine tuned to generate a question with answers from a context , why that can be very usful this can help you to generate a dataset from a book article any thing you would to make from it dataset and train another model on this dataset , give the model any context with pre prometed of quation you want + context and it will extarct question + answer for you 
this are promted i use
>[ "which", "how", "when", "where", "who", "whom", "whose", "why",
    "which", "who", "whom", "whose", "whereas",
    "can", "could", "may", "might", "will", "would", "shall", "should",
    "do", "does", "did", "is", "are", "am", "was", "were", "be", "being", "been",
    "have", "has", "had", "if", "is", "are", "am", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "shall", "should", "might", "may", "must",
    "may", "might", "must"]
>
#  orignal model info
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/flan2_architecture.jpg"
alt="drawing" width="600"/>

#  Table of Contents

0. [TL;DR](#TL;DR)
1. [Model Details](#model-details)
2. [Usage](#usage)
3. [Uses](#uses)
4. [Bias, Risks, and Limitations](#bias-risks-and-limitations)
5. [Training Details](#training-details)
6. [Evaluation](#evaluation)
7. [Environmental Impact](#environmental-impact)
8. [Citation](#citation)
9. [Model Card Authors](#model-card-authors)

# TL;DR

If you already know T5, FLAN-T5 is just better at everything. For the same number of parameters, these models have been fine-tuned on more than 1000 additional tasks covering also more languages. 
As mentioned in the first few lines of the abstract : 
>  Flan-PaLM 540B achieves state-of-the-art performance on several benchmarks, such as 75.2% on five-shot MMLU. We also publicly release Flan-T5 checkpoints,1 which achieve strong few-shot performance even compared to much larger models, such as PaLM 62B. Overall, instruction finetuning is a general method for improving the performance and usability of pretrained language models.
# Model Details

## Model Description


- **Model type:** Language model
- **Language(s) (NLP):** English
- **License:** mit 
- **Related Models:** [All FLAN-T5 Checkpoints](https://huggingface.co/models?search=flan-t5)
- **Original Checkpoints:** [All Original FLAN-T5 Checkpoints](https://github.com/google-research/t5x/blob/main/docs/models.md#flan-t5-checkpoints)
- **Resources for more information:**
  - [Research paper](https://arxiv.org/pdf/2210.11416.pdf)
  - [GitHub Repo](https://github.com/google-research/t5x)
  - [Hugging Face FLAN-T5 Docs (Similar to T5) ](https://huggingface.co/docs/transformers/model_doc/t5)

# Usage

Find below some example scripts on how to use the model in `transformers`:

## Using the Pytorch model

### Running the model on a CPU

<details>
<summary> Click to expand </summary>

```python

from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("mohamedemam/QA_GeneraToR")
model = T5ForConditionalGeneration.from_pretrained("mohamedemam/QA_GeneraToR")

input_text = r"when: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards[note 2] and a record six European Golden Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving the club"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU

<details>
<summary> Click to expand </summary>

```python
# pip install accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("mohamedemam/QA_GeneraToR")
model = T5ForConditionalGeneration.from_pretrained("mohamedemam/QA_GeneraToR", device_map="auto")

input_text = r"when: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards[note 2] and a record six European Golden Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving the club"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

### Running the model on a GPU using different precisions

#### FP16

<details>
<summary> Click to expand </summary>

```python
# pip install accelerate
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("mohamedemam/QA_GeneraToR")
model = T5ForConditionalGeneration.from_pretrained("mohamedemam/QA_GeneraToR", device_map="auto", torch_dtype=torch.float16)

input_text = r"when: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards[note 2] and a record six European Golden Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving the club"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

#### INT8

<details>
<summary> Click to expand </summary>

```python
# pip install bitsandbytes accelerate
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("mohamedemam/QA_GeneraToR")
model = T5ForConditionalGeneration.from_pretrained("mohamedemam/QA_GeneraToR", device_map="auto", load_in_8bit=True)

input_text = r"when: Lionel Andrés Messi[note 1] (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (listen); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards[note 2] and a record six European Golden Shoes, and in 2020 he was named to the Ballon d'Or Dream Team. Until leaving the club"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

</details>

# Uses

## Direct Use and Downstream Use

The authors write in [the original paper's model card](https://arxiv.org/pdf/2210.11416.pdf) that: 

> The primary use is research on language models, including: research on zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models

See the [research paper](https://arxiv.org/pdf/2210.11416.pdf) for further details.

## Out-of-Scope Use

More information needed.

# Bias, Risks, and Limitations

The information below in this section are copied from the model's [official model card](https://arxiv.org/pdf/2210.11416.pdf):

> Language models, including Flan-T5, can potentially be used for language generation in a harmful way, according to Rae et al. (2021). Flan-T5 should not be used directly in any application, without a prior assessment of safety and fairness concerns specific to the application.

## Ethical considerations and risks

> Flan-T5 is fine-tuned on a large corpus of text data that was not filtered for explicit content or assessed for existing biases. As a result the model itself is potentially vulnerable to generating equivalently inappropriate content or replicating inherent biases in the underlying data.

## Known Limitations

> Flan-T5 has not been tested in real world applications.

## Sensitive Use:

> Flan-T5 should not be applied for any unacceptable use cases, e.g., generation of abusive speech.

# Training Details

## Training Data

The model was trained on a mixture of tasks, that includes the tasks described in the table below (from the original paper, figure 2):

![table.png](https://s3.amazonaws.com/moonup/production/uploads/1666363265279-62441d1d9fdefb55a0b7d12c.png)


## Training Procedure

According to the model card from the [original paper](https://arxiv.org/pdf/2210.11416.pdf):

> These models are based on pretrained T5 (Raffel et al., 2020) and fine-tuned with instructions for better zero-shot and few-shot performance. There is one fine-tuned Flan model per T5 model size.

The model has been trained on TPU v3 or TPU v4 pods, using [`t5x`](https://github.com/google-research/t5x) codebase together with [`jax`](https://github.com/google/jax).


# Evaluation

## Testing Data, Factors & Metrics

The authors evaluated the model on various tasks covering several languages (1836 in total). See the table below for some quantitative evaluation:
![image.png](https://s3.amazonaws.com/moonup/production/uploads/1668072995230-62441d1d9fdefb55a0b7d12c.png)
For full details, please check the [research paper](https://arxiv.org/pdf/2210.11416.pdf).

## Results 

For full results for FLAN-T5-Large, see the [research paper](https://arxiv.org/pdf/2210.11416.pdf), Table 3.

# Environmental Impact

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** Google Cloud TPU Pods - TPU v3 or TPU v4  | Number of chips ≥ 4.
- **Hours used:** More information needed
- **Cloud Provider:** GCP
- **Compute Region:** More information needed
- **Carbon Emitted:** More information needed

# Citation

**BibTeX:**

```bibtex
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  
  url = {https://arxiv.org/abs/2210.11416},
  
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scaling Instruction-Finetuned Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
```
