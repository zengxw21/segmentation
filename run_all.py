import os

def run_all():
    learning_rate_list=[0.001]
    batch_size_list=[50]
    max_length_list=[120]
    for rate in learning_rate_list:
        for bs in batch_size_list:
            for max_length in max_length_list:
                print("RNN_LSTM")
                os.system(f"python main.py -n RNN_LSTM -l {rate} -b {bs} -m {max_length}")
                print("LSTM_Attention")
                os.system(f"python main.py -n LSTM_Attention -l {rate} -b {bs} -m {max_length}")
                print("RNN_GRU")
                os.system(f"python main.py -n RNN_GRU -l {rate} -b {bs} -m {max_length}")
                print("textCNN")
                os.system(f"python main.py -n textCNN -l {rate} -b {bs} -m {max_length}")
                print("MLP")
                os.system(f"python main.py -n MLP -l {rate} -b {bs} -m {max_length}")
                #print("transformer")
                #os.system(f"python main.py -n transformer -l {rate} -b {bs} -m {max_length}")



if __name__=="__main__":
    run_all()
