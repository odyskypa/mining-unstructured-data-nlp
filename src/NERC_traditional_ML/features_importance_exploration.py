import pycrfsuite
import pandas as pd
import sys
import os

def print_features(tagger, label):
    # Get the state features for the given label
    state_features = {k: v for k, v in tagger.info().state_features.items() if k[1] == label}

    # Sort the state features by absolute value of score
    sorted_features = sorted(state_features.items(), key=lambda x: abs(x[1]), reverse=True)

    # Print the top 10 features
    print(f"Sorted features for label {label}:")
    features = []
    scores = []
    # for i, (feat, score) in enumerate(sorted_features[:10]):
        # print(f"{i+1}. {feat} ({score:.4f})")
    for i, (feat, score) in enumerate(sorted_features):
        # features.append(feat)
        scores.append(abs(score))
        if "=" in feat[0]:
            f = feat[0].split('=')[0]
        else:
            f = feat[0]
        features.append(f)
    df = pd.DataFrame({'features': features, 'scores': scores})
    return df


if __name__ == '__main__':

    # get file where model will be written
    modelfile = sys.argv[1]

    # Open the model file for reading
    tagger = pycrfsuite.Tagger()
    tagger.open(modelfile)

    # Get the labels from the model
    labels = tagger.labels()

    dir_name = 'feature_csvs_' + modelfile

    if not os.path.exists(dir_name):
        # Create the directory if it doesn't exist
        os.mkdir(dir_name)


    # Print the top features for each label
    for label in labels:
        df = print_features(tagger, label)
        df.to_csv(f"{dir_name}/dev_features_of_label_{label}.csv", index=False)