# Train and test on waveform with pitch detection model
#python ".\conv1d\train_wav_pitch.py" -data "F:\Datasets\TimbreTransfer"
#python ".\conv1d\eval_wav_pitch.py" -data "F:\Datasets\TimbreTranfer" -model "trained_models/latest_pitch_model_9.pt" -out "F:\Datasets\TimbreTranfer\output\wave_pitch"

# Train and test on rnn model
python ".\rnn\train_rnn.py" -data "F:\Datasets\TimbreTransfer"
python ".\rnn\train_rnn.py" -data "F:\Datasets\TimbreTranfer" -model "trained_models/latest_rnn_model_30.pt" -out "F:\Datasets\TimbreTranfer\output\rnn"