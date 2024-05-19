# Contextual-Emotion-Recognition

**Generating Narracap:**

1. Create the environment:
```bash
cd narracap
conda create -n narracap python=3.10 -y
conda activate narracap
pip install -r requirements.txt
```

2. Run the following:
```bash
python main.py --image_BB "path/to/images/with/bb" --image_cropped "path/to/cropped/bb/of/target" --emotic "path/to/emotic"
```


**Evaluation(bootstrap):**
```bash
cd evaluation
python bootstarp.py --input "input questions and IDs to LLaVA" --output "LLaVA responses"
```
