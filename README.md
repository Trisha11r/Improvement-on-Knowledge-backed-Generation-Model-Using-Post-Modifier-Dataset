# Improvement-on-Knowledge-backed-Generation-Model-Using-Post-Modifier-Dataset (POMO)

- Post-modifier is a short descriptive phrase that comes after a word (usually a noun) in a sentence which gives more detailed contextual information about that word. Post modifier generation model has two sub tasks - selecting a set of facts or claims about the concerned entity (WikiData) and from that set selecting the most relevant fact to generate a post modifier that contextually describes it.(POMO Dataset)

- We trained the different neural models for 100 and 800 epochs respectively, and the results for both the cases deemed the model using bi- Directional LSTM encoder/decoder module to be the best performing one, in terms of Prediction Score. Thus, for this model we try to improve the PoMo by altering the attention mechanism for the model. The Global attention function (Luong) gave us the optimal results.

**Model Architecture**: Bi-LSTM (2 layer) model with attention function

So, we had two datasets: 
      - WikiData: Entity and its facts
      - POMO Dataset: Sentence, Entity, Post Modifier (Context - Prev & Next Sentence)

**File pm_generation.py:**

There are 4 choices in encoder_type such as RNN, biRNN, mean and transformer.
Similarly, we used the 3 choices available in rnn_type, namely, LSTM, GRU and SRU.

## Commands Required:
This model architecture uses the in-built Open NMT system to build the model.

**Step (1)- Set the Open NMT System:** 

Run the command:
      - pip install OpenNMT-py
      - Write the following in the util.py
      - OpenNMT_dir = "OpenNMT_location" (to use the in-built OpenNMT System)


**Step (2)- Preprocess Data:**

python pm_generation.py prepare -data_dir dataset_location -data_out prefix

This is using preprocess.py in OpenNMT system, and will genereate three following files:

      (prefix).train.pt: Pytorch file containing training data
      (prefix).valid.pt: Pytorch file containing validation data
      (prefix).vocab.pt: Pytorch file containing vocabulary data

When running this command, it will create 2 temp files- source and target. The source file stores the sentences(from .pm file) + claims( from .wiki file) and the target file contains the post modifiers (from the .pm file). This is how data is preprocessed so that our model can be trained as having sentences+claims as features and post modifiers as labels.

**Step (3) Training:**

python pm_generation.py train -data prefix -model model_location

This is using train.py in OpenNMT system. The model consists of a 2-layer biLSTM with 500 hidden units on the encoder and a 2-layer LSTM with 500 hidden units on the decoder. Attention is used the general scheme, which is a multiplicative global attention with one weight matrix.

This step basically trains the model on preprocessed data to get relevancy score in the training the model.

**Step (4) Generate Post-modifier:**

python pm_generation.py generate -data_dir dataset_location -dataset dataset_prefix -model model_dir -out output_file

This is using translate.py in OpenNMT system which uses the model directory mentioned in the command line arguments to generate the post modifiers.

## Software Requirements:

Run the command:
pip install -r OpenNMT-py/requirements.opt.txt


requirements.opt.txt (this file has the following requirments)

            tensorflow 2.4
            cffi
            torchvision 
            joblib
            librosa
            Pillow
            git+git://github.com/pytorch/audio.git@d92de5b97fc6204db4b1e3ed20c03ac06f5d53f0
            pyrouge
            opencv-python
            git+https://github.com/NVIDIA/apex
            pretrainedmodels


## Original Source for code base:

(1) https://github.com/StonyBrookNLP/PoMo
(2) https://github.com/StonyBrookNLP/PostModifierGenerati

## Model
Link : https://drive.google.com/drive/folders/11bhYkI8wHrukBbqf0wok_DdQCAildFeW?usp=sharing

## Dataset
Contact: 
- shanuj.shekhar@stonybrook.edu
- trisha.kanji@stonybrook.edu
- priyanka.nath@stonybrook.edu
      
