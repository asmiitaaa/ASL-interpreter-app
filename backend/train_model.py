import pandas as pd
import pickle#saves the trained model as a .pkl file 
import os
from sklearn.ensemble import RandomForestClassifier 


DATA_FILE='data/asl_data.csv'
MODEL_FILE='models/asl_model.pkl'

def load_data():
    df=pd.read_csv(DATA_FILE)#loading the csv file into a pandas obj 

    X=df.drop('label', axis=1).values
    y=df['label'].values#answers 

    print(f"Loaded {len(X)} samples")
    print(f"Signs found:{sorted(set(y))}")

    return X,y

def train(X,y):
    print("\n Training the model...")

    model = RandomForestClassifier(
        n_estimators=100,#100 decision trees vote on each prediction 
                                 random_state=42 )
    
    #randomness we can set as any constant, just taking 42 

    model.fit(X, y)

    return model 

def save_model(model, labels):
    #create models folder if it doesnt exist
    os.makedirs('models', exist_ok=True)

    # save model and labels together in one file
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({'model': model, 'labels': labels}, f)

    print(f"\nModel saved to {MODEL_FILE}")


def main():
    #Loading the data 
    X,y=load_data()

    #train on all of the loaded data 
    model=train(X,y)

    #now, we save the model 
    labels=sorted(set(y))
    save_model(model, labels)

    print("\nDone! Ready to test live!")


if __name__ == '__main__':
    main()
#just runs the main if we execute the file directly 
