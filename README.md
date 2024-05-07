# LLM - Detect AI-Generated Text

Develop a machine learning model capable of identifying whether a piece of text was written by a student or produced by an LLM.

### Exploring:
- GPT2.ipynb (We explored the GPT2 model and decided not to use in zero shot classification)

### Zero-shot classification:
- bloom.ipynb - Zero shot classification for Bloom
- MistralAi-Guidance-Genre-ZeroShot.ipynb - Zero shot classification for Mistral
- f_llama2_genre.py -Zero shot genre based classification for Llama2
- f_llama2_genre.py - Zero shot classification for Llama2

### Finetuning:
- f_llama2_genre.py - Llama2 for genre based classification
- f_llama2_genre.py - Llama2 for test classification
- Fine_Tuning_LLM_and_Prediction_MISTRAL.ipynb - Mistral fine-tuning

In case of Llama2, we used the same code across zero shot and final predictions, but loaded the fine-tuned model for test predictions.




