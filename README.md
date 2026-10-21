# Tennis Stroke Classification from Video

Computer vision project that classifies tennis video clips into four classes: **Forehand, Backhand, Serves, NoStroke**.

## Approach
`TENNIS.ipynb` builds the model step by step:
1. Baseline CNN
2. A second model
3. Final model: pretrained **ResNet-18** frame features fed to an **LSTM** over a sequence of frames, trained for longer in a second version

`EVAL_TENNIS.ipynb` loads the final model and evaluates it on held-out test videos.

## Results (15 test videos)
| Metric | Score |
|---|---|
| Accuracy | 0.933 |
| Macro F1 | 0.942 |

## Running
The notebooks were run in Google Colab with the video dataset on Google Drive (not included).
Main libraries: `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `matplotlib`, `seaborn`, `plotly`.
