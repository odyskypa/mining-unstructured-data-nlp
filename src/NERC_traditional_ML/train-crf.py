#!/usr/bin/env python3
import time
import pycrfsuite
import sys
from contextlib import redirect_stdout

def instances(fi):
    xseq = []
    yseq = []
    
    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line means the end of a sentence.
            # Return accumulated sequences, and reinitialize.
            yield xseq, yseq
            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')
        
        # Append the item features to the item sequence.
        # fields are:  0=sid, 1=form, 2=span_start, 3=span_end, 4=tag, 5...N = features
        item = fields[5:]        
        xseq.append(item)
        
        # Append the label to the label sequence.
        yseq.append(fields[4])


if __name__ == '__main__':

    # get file where model will be written
    modelfile = sys.argv[1]
    
    # Create a Trainer object.
    trainer = pycrfsuite.Trainer()
    
    # Read training instances from STDIN, and append them to the trainer.
    for xseq, yseq in instances(sys.stdin):
        trainer.append(xseq, yseq, 0)

    # Initialize the training algorithm. Use L2-regularized SGD and 1st-order dyad features.
    trainer.select('l2sgd', 'crf1d')

    # Set a training parameter. This demonstrates how to list parameters and obtain their values.
    trainer.set('feature.minfreq', 1)  # mininum frequecy of a feature to consider it
    trainer.set('c2', 0.25)  # coefficient for L2 regularization
    trainer.set('feature.possible_states', False)  # shit
    trainer.set('feature.possible_transitions', False)
    trainer.set('max_iterations', 1000)# shit doesn't do
    #trainer.set('delta', 1e-06) # threshold for the stopping criterion, which is based on the improvement of the log-likelihood over the last period iterations
    #trainer.set('period', 20) # duration of iterations to test the stopping criterion. A higher value of period may result in slower convergence

    #for learning rate calibration, if more thorough or not
    #trainer.set('calibration.eta', 0.05)
    #trainer.set('calibration.rate', 1.5)
    #trainer.set('calibration.samples', 500)
    #trainer.set('calibration.candidates', 5)
    #trainer.set('calibration.max_trials', 10)


    print("Training with following parameters: ")
    for name in trainer.params():
        print (name, trainer.get(name), trainer.help(name), file=sys.stderr)

    st = time.time()

    # Start training and dump model to modelfile
    trainer.train(modelfile, -1)

    # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    sys.stderr.write(f'Execution time: {elapsed_time} seconds\n')
