# Contrastive Language–Image Pre-training

## What this is about
Just a simple implementation based on the CLIP which seems to be an important thing especially when we talk about generative AI, and zero-shot learning 

## What has been done 
1. Set up the Architecture
1. Set up loss function
1. Set up the dataset and dataloader 
1. Set up the training, which could be better implemented admittedly.
1. Set up validation to get validation loss.

## What else needs to be done

1. Inference Pipeline (encode and embeddings)
1. Results visualisation

## How to run 

Make sure you change the directory of your data. I used Flickr dataset - 
```
python -m src.main
```

## Visualisation

## Resources

### From scratch implementation

https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py - when the CLIP implementation by OPENAI didn't make sense to me, I looked at this
https://github.com/openai/CLIP/blob/main/clip/model.py - the original. Quite hard to read honestly.

