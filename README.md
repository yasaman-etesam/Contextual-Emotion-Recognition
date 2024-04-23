# Contextual-Emotion-Recognition

# Generating Narracap

```bash
conda create -n narracap python=3.10 -y
conda activate narracap
pip install -r requirements.txt


To generate captions:
1. Create conda env:
     conda create -n narracap python=3.10 -y
      conda activate narracap

2. pip install -r requirements.txt
3. python main.py --image_BB "path/to/images/with/bb" --image_cropped "path/to/cropped/bb/of/target" --emotic "path/to/emotic"
