import numpy as np
import csv


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def least_squares_GD(y, tx, initial_w,max_iters, gamma):
    """
    Linear regression using gradient descent and least squares
    """
    N, D = tx.shape
    
    # Iterations of gradient descent
    w = initial_w
    for _ in range(max_iters):
        grad = -np.dot(tx.T, (y - np.dot(tx,w))) / N
        w = w - gamma * grad
        
    # Calculating the loss
    r = y - np.dot(tx,w)
    loss = np.dot(r,r) / (2*N)
    
    return w, loss

#Linear regression using stochastic gradient descent
def least_squares_SGD(y, tx, initial_w,max_iters, gamma, frequency=0):
    """Linear regression using stochastic gradient descent and least squares"""
    N, D = tx.shape
    
    # Iterations of stochastic gradient descent
    w = initial_w
    for i in range(max_iters):
        k = np.random.randint(0,N-1)
        grad = -(y[k]-np.dot(tx[k,:], w))*tx[k,:]
        w = w - gamma * grad
        
            
    r = y - np.dot(tx,w)
    loss = np.dot(r,r) / (2*N)    
    
    return w, loss

#Least squares regression using normal equations
def least_squares(y, tx):
    N, _ = tx
    
    # Calculating w
    w = (np.linalg.inv((tx.T).dot(tx)).dot(tx.T)).dot(y)
    
    #Calculating loss
    r = y - tx.dot(w)
    loss = np.dot(r,r)/(2*N)
    return w, loss

# Returns 1/(1+exp(-x))
# x is scalar or numpy array
def sigmoid(x):
    tmp = np.exp(-x)
    return 1/(1+tmp)




def logistic_regression(y, tx, initial_w,max_iters, gamma):
    """
    #Logistic regression using SGD
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    # initial_w:  vector (dimension D)
    # max_iters:  scalar
    # gamma:      scalar respresenting step size
    # return parameters w for the regression and loss
    """
    return reg_logistic_regression(y, tx, 0, initial_w,max_iters, gamma)


def logistic_regression_GD(y, tx, initial_w,max_iters, gamma):
    """
    # Logistic regression using GD
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    # initial_w:  vector (dimension D)
    # max_iters:  scalar
    # gamma:      scalar respresenting step size
    # return parameters w for the regression and loss
    """
    return reg_logistic_regression_GD(y, tx, 0, initial_w,max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
    #Regularized logistic regression using SGD
    # y:          vector of outputs (dimension N)
    # tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    # lambda:     scalar representing regularization parameter
    # initial_w:  vector (dimension D)
    # max_iters:  scalar
    # gamma:      scalar respresenting step size
    # return parameters w for the regression and loss
    """
    N, _ = tx.shape
    w = initial_w
    
    
    for i in range(max_iters):
        k = np.random.randint(0,N-1)
        tmp = np.dot(tx[k,:],w)
        grad = -y[k]*tx[k,:]+sigmoid(tmp)*tx[k,:]+lambda_*w
        w = np.squeeze(np.asarray(w - gamma*grad))
        
        
        
    tmp = np.squeeze(np.asarray(np.dot(tx,w)))
    loss = - np.dot(tmp, y.T)
    loss += np.sum(np.log(1+np.exp(tmp)))
    loss /= N

    return w, loss

def reg_logistic_regression_GD(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """
    Regularized logistic regression using GD
    y:          vector of outputs (dimension N)
    tx:         matrix of data (dimension N x D), such that tx[:, 0] = 1
    lambda:     scalar representing regularization parameter
    initial_w:  vector (dimension D)
    max_iters:  scalar
    gamma:      scalar respresenting step size
    return parameters w for the regression and loss
    """

    w = initial_w
    
    for _ in range(max_iters):
        tmp = np.dot(tx, w)
        grad = np.dot((sigmoid(tmp) - y), tx) + lambda_*w
        w = w - gamma*grad
        w = np.squeeze(np.asarray(w))
    
    tmp = np.dot(tx,w)
    tmp = np.squeeze(np.asarray(tmp))
    loss = - np.dot(y,tmp)+np.sum(np.log(1+np.exp(tmp)))
    return w, loss

def add_column(data, column):
    tmp = np.asmatrix(column).T
    return np.concatenate((data, tmp), axis=1)


def transform_output(y):
    """
    Function that transforms prediction column, such that all -1 values are replaced with 1
    and all 1 values are replaced with 0
    """
    return np.array(y == -1, np.int) 
    



def transform_categorical_data(train_data, test_data, column_names):
    """
    Function that transforms categorical data by droping the column 'PRI_jet_num' 
    and adding 4 new columns 'PRI_jet_num{i}' that represents indicator if 'PRI_jet_num' equals i
    """

    # Locating index of PRI_jet_num column
    for i in range(len(column_names)):
        if column_names[i] == 'PRI_jet_num':
            cat_id = i

    # Adding new columns
    for i in range(4):
        column_names = np.append(column_names,  f'PRI_jet_num{i}')
        new_column_train = np.array([int(x==i) for x in train_data[:,cat_id]])
        new_column_test = np.array([int(x==i) for x in test_data[:,cat_id]])
        train_data = np.concatenate((train_data, np.asmatrix(new_column_train).T), axis=1)
        test_data = np.concatenate((test_data, np.asmatrix(new_column_test).T), axis=1)

    # Droping PRI_jet_num column
    column_names = np.delete(column_names, cat_id)
    train_data = np.delete(train_data, cat_id, axis=1)
    test_data = np.delete(test_data, cat_id, axis=1)

    return train_data, test_data, column_names

 
def fill_nans_with_median(train_data, column_names):
    """
    Function that replaces all the nans (in our case -999) with median of that column (excluding -999)
    """
    # Finding all the columns that contain -999
    columns_with_nan = []
    for i in range(len(column_names)):
        if train_data[:,i].min() == -999:
            columns_with_nan.append(i)
    
    # Replacing -999 with median
    for i in columns_with_nan:
        column = train_data[:,i]
        median = np.median(np.squeeze(np.array(column[column!=-999])))
        column[column==-999] = median

    return train_data


def standardize(data, column_names):
    """
    Function that standardize data by applying linear transformation such that minimum of each column 
    is 0 and maximum is 1, except from constant columns
    """
    # We ignore standardization if column is a constant
    for i in range(data.shape[1]):
        if column_names[i] == 'Constant':
            continue
        data[:,i] = (data[:,i]-data[:,i].min())/(data[:,i].max()-data[:,i].min())
    return data


def feature_exp_expansion(data, column_names, target_columns):
    """
    For every index i in the list target_columns, exponential of i-th column is added to data
    If i is an index of categorical column, nothing is added
    """
    new_columns = np.copy(column_names)
    for i in target_columns:
        # Checking if a column has categorical values
        if column_names[i] in ['PRI_jet_num0', 'PRI_jet_num1', 'PRI_jet_num2', 'PRI_jet_num3', 'Prediction']:
            continue
       
        # Adding new column
        new_column = np.exp(data[:,i])
        data = np.concatenate((data, new_column), axis=1)
        new_columns = np.append(new_columns, f'exp_{column_names[i]}')

    return data, new_columns
            

def feature_log_expansion(data, column_names, target_columns):
    """
    For every index i in the list target_columns, logarithm of i-th column is added to data
    If i-th column is categorical or contains negative values or 0, nothing is added
    """
    for i in target_columns:
        # Checking if a column has categorical values
        if column_names[i] in ['PRI_jet_num0', 'PRI_jet_num1', 'PRI_jet_num2', 'PRI_jet_num3', 'Prediction']:
            continue
       
        # Checking if a column contains negative values or 0
        if data[:,i].min() <= 0:
            continue
        
        # Adding new column
        new_column = np.log(data[:,i])
        data = np.concatenate((data, new_column), axis=1)
        column_names = np.append(column_names, f'log_{column_names[i]}')

    return data, column_names


def feature_polynomial_expansion(data, column_names, target_column, degree):
    """
    For every index i in the list target_columns, first degree degrees of i-th column are added to data
    If i-th column is categorical nothing is added
    if degree equals 0, columns are dropped
    """
    # Making a copy of data
    new_data = np.copy(data)
    new_column_names = np.copy(column_names)

    # If degree equals 1, nothing is added
    if degree == 1:
        return new_data, new_column_names
    
    # If degree equals 0, specified columns are dropped
    if degree == 0:
        new_data = np.delete(data, target_column, axis=1)
        new_column_names = np.delete(new_column_names.remove, target_column)
        return new_data, new_column_names
    
    # if degree is greater than 1 non-categorical columns are added
    if column_names[target_column] in ['PRI_jet_num0', 'PRI_jet_num1', 'PRI_jet_num2', 'PRI_jet_num3', 'Prediction', 'Constant']:
        return new_data, new_column_names
    for d in range(2,degree+1):
        new_column = new_data[:,[target_column]] ** d
        data = np.concatenate((data, new_column), axis=1)
        new_column_names = np.append(new_column_names, f'{column_names[target_column]}^{d}')
            
    return new_data, new_column_names


def add_constant_attribute(data, column_names):
    """Function that adds constant column"""
    N, _ = data.shape
    new_data = np.copy(data)
    new_data = add_column(new_data, np.ones(N))
    column_names = np.append(column_names, 'Constant')
    return new_data, column_names


def split (y, input_data, column_names, train_ratio=0.50, seed=42):
    """Function that splits data into train and test data such that training data is balanced"""
    np.random.seed(seed)

    # Merging output to input so that we can preform shuffle
    data = np.concatenate((input_data, np.asmatrix(y).T), axis=1)
   
    
    # We do not allow ratio of training and testing set to be larger than 0.66 
    # in order to have training set with same number of 0 and 1 values
    TRAINING_TRESHOLD = 0.66
    if train_ratio > TRAINING_TRESHOLD:
        train_ratio = TRAINING_TRESHOLD
    train_size = round(train_ratio * len(data))
    
    # Dividing data into bosons and non bosons
    prediction_id = data.shape[1] - 1
    prediction = np.squeeze(np.asarray(data[:,prediction_id]))
    is_boson = (prediction == 1)
    boson = data[is_boson]
    is_not_boson = (prediction == 0)
    spiner = data[is_not_boson]

    # Randomizing the order of slices and taking 
    np.random.shuffle(boson)
    np.random.shuffle(spiner)
    train = np.concatenate((boson[:round(train_size/2)], spiner[:round(train_size/2)]))
    test = np.concatenate((boson[round(train_size/2):], spiner[round(train_size/2):]))

    # Spliting train and test into input and output
   
    y_train = np.squeeze(np.asarray(train[:,prediction_id]))
    y_test = np.squeeze(np.asarray(test[:,prediction_id]))      
    x_train = np.delete(train, prediction_id, axis=1)
    x_test = np.delete(test, prediction_id, axis=1)
   
    return x_train, y_train, x_test, y_test, column_names


def predict(x, w):
    """ Given data x and model w, predict y using logistic regression """
    p = np.squeeze(np.asarray(sigmoid(np.dot(x,w))))
    
    return np.array([int(i>0.5) for i in p])

def get_accuracy(y,y_predict):
    return 1 - np.sum(abs(y-y_predict))/ len(y)     

def get_recall(y,y_predict):
    return np.count_nonzero(np.logical_and(y,y_predict)) / np.count_nonzero(y)

def get_precision(y,y_predict):
    if np.count_nonzero(y_predict) == 0:
        return 0
    return np.count_nonzero(np.logical_and(y,y_predict)) / np.count_nonzero(y_predict)

def evaluate_on_set(w, x, y):
    y_predict = predict(x, w)
    accuracy = get_accuracy(y,y_predict)
    recall = get_recall(y,y_predict)
    precision = get_precision(y,y_predict)
    if recall+precision == 0:
        return {'accuracy' : accuracy, 'recall': recall, 'precision':precision, 'f1_score':0}
    f1_score = 2*recall*precision/(recall+precision)
    return {'accuracy' : accuracy, 'recall': recall, 'precision':precision, 'f1_score':f1_score}



def run(y, data, column_names, hyperparameters, GD=False, train_ratio=0.5):
    """
    Given labeled data and set of hyperparameters, this function:
    adds polynomial attributes
    splits the data into training and test set
    trains the model using logistic regression that is based on GD (if GD=True) or SGD (if GD=False)
    Function returns obtained set of parameters w, evaluation on the training and test set
    """
    curr_data= np.copy(data)
    
    curr_data = transform_data(curr_data, column_names, hyperparameters)

    seed = np.random.randint(1,100)
    x_train, y_train, x_test, y_test, column_names = split(y, curr_data, column_names, seed=seed, train_ratio=train_ratio)
    max_iters = hyperparameters['max_iters']
    lambda_ = hyperparameters['lambda']
    gamma = hyperparameters['gamma']
    initial_w = np.squeeze(np.asarray(x_train.mean(axis=0)))
    
    if GD:
        w, loss = reg_logistic_regression_GD(y_train, x_train, lambda_ ,initial_w, max_iters, gamma)
    else:
        w, loss = reg_logistic_regression(y_train, x_train, lambda_ ,initial_w, max_iters, gamma)
                                  
    evaluation_test = evaluate_on_set(w, x_test, y_test)
    evaluation_train = evaluate_on_set(w, x_train, y_train)
    
    return w, evaluation_test, evaluation_train


def make_population(population_size, column_names, max_iters):
    """
    Implementation of genetic algorithm and necessery functions
    """
    population = []
    for i in range(population_size):
        lambda_ = np.random.choice([0, 0.00001, 0.00005, 0.0001,0.0005,0.001])
        gamma= np.random.choice([0.00001, 0.00005, 0.0001])
        param = {'lambda' : lambda_, 'gamma' : gamma, 'max_iters' : max_iters}
        for key in column_names:
            if 'PRI_jet_num' in key:
                param[key] = np.random.randint(1,2)
            else:
                param[key] = np.random.randint(1,4)
        population.append(param)
    return population


def make_offspring(parent1, parent2, mutation_rate=0.05):
    """ Function that produces two new child chromosomes from parent chromosomes """
    n = len(parent1)
    baby1 = {}
    baby2 = {}
    crossing_point = np.random.randint(1,n-1)
    for i,key in enumerate(parent1.keys()):
        if key == 'max_iters':
            baby1[key], baby2[key] = parent1['max_iters'], parent1['max_iters']
            continue

        mutation_indicator = np.random.uniform(0,1)
        if mutation_indicator > mutation_rate:
            if i > crossing_point:
                baby1[key] = parent1[key]
                baby2[key] = parent2[key]
            else:
                baby2[key] = parent1[key]
                baby1[key] = parent2[key]
        else:
            if key =='lambda':
                baby2[key] = np.random.choice([0, 0.00001, 0.00005, 0.0001,0.0005,0.001])
                baby1[key] =  np.random.choice([0, 0.00001, 0.00005, 0.0001,0.0005,0.001])
            elif key == 'gamma':
                baby2[key] = np.random.choice([0.00001, 0.00005, 0.0001])
                baby1[key] = np.random.choice([0.00001, 0.00005, 0.0001])
            elif 'PRI_num_jet' in key:
                baby2[key] = np.random.choice([1])
                baby1[key] = np.random.choice([1])
            else:
                baby2[key] = np.random.choice([1, 2, 3])
                baby1[key] = np.random.choice([1, 2, 3])

    return baby1, baby2


def take_best_from_population(population, fitness, size):
    """Taking the size best chromosomes from a population based on fitnesses"""
    sorted_population = sorted(population, key=lambda x: -fitness[str(x)])
    return sorted_population[:size]

def optimize(y, train_data, column_names, GD, population_size=20, size=10, max_iters=40, log=None):
    """  FUnction that searches for the best hyperparameters by using Genetic algorithm """
    # Openning a file for storing the results
    if log != None:
        f = open(log, 'a')

    # Generating population randomly
    max_iters = 200 if GD else 10
    population = make_population(population_size, column_names, max_iters) 

    # Dictionary that stores fitness of each hyperparameter configuration  
    # Fitness as calculated as minimum of test and training accuracy
    # Keys of the dictionary are strings of hyperparameters
    fitness = {}

    # Dictionary that stores parameters obtained by training the model 
    # for each hyperparameter configuration
    # Keys of the dictionary are strings of hyperparameters
    w = {}

    for i in range(max_iters):
        for param in population:
            if str(param) in fitness:
                continue
            # Training the model and setting fitness
            w[str(param)], test_evaluation, train_evaluation = run(y, train_data, column_names, param, GD)
            fitness[str(param)] = min(test_evaluation['accuracy'], train_evaluation['accuracy'])

            # Writing obtained results in a file
            if log != None:
                f.write(str(w[str(param)]))
                f.write('\n')
                f.write(str(param))
                f.write('\n')
                f.write(str(fitness[str(param)]))
                f.write('\n-----------------------------------------------\n')
            
        # Chosing the best instances from population and pair them randomly
        best = take_best_from_population(population, fitness, size)
        np.random.shuffle(best)
        for i in range(len(best)//2):
            baby1, baby2 = make_offspring(best[2*i], best[2*i+1])
            best.append(baby1)
            best.append(baby2)
            
        population = best
    
    if log != None:
        f.close()
    
    return fitness, w


def get_parameters_from_str(s):
    """ Function that recreates dictionary of hyperparameters from string """
    arr = s.split(',')
    param = {}
    for i in range(0, len(arr)):
        j1 = arr[i].find('\'')+1
        j2 = arr[i][j1:].find('\'')
        key = arr[i][j1:][:j2]
        i1 = arr[i].find(':')+1
        if arr[i].find('}') > 0:
            arr[i] = arr[i][:-1]
        if i < 2:
            num = float(arr[i][i1:])
        else:
            num = int(arr[i][i1:])
        param[key] = num
    return param


def log_exp_feature_augmentation_and_standardization(train_data, test_data, column_names):
    """
    Function that does preprocessing by adding exponential, logarithmic and constant features 
    and standardizing data on 0 1
    """
    

    # Making a list of all attributes that are strictly positive
    non_categorical_columns = range(len(column_names)-4)
    positive_columns = []
    for column in non_categorical_columns:
        if train_data[:,column].min() > 0 and test_data[:,column].min()>0:
            positive_columns.append(column)

    # Adding logarithmic column
    old_columns = np.copy(column_names)
    train_data, column_names = feature_log_expansion(train_data, old_columns, positive_columns)
    test_data, _ = feature_log_expansion(test_data, old_columns, positive_columns)

    # Standardizing data before adding exponential columns in order to avoid overflow
    train_data = standardize(train_data, column_names)
    test_data = standardize(test_data, column_names)

    # Adding exponential columns for all non categorical data
    old_columns = np.copy(column_names)
    train_data, column_names = feature_exp_expansion(train_data, old_columns, non_categorical_columns)
    test_data, _ = feature_exp_expansion(test_data, old_columns, non_categorical_columns)

    # Standardizing data
    train_data = standardize(train_data, column_names)
    test_data = standardize(test_data, column_names)

    # Adding constant attribute
    old_columns = np.copy(column_names)
    train_data, column_names = add_constant_attribute(train_data, old_columns)
    test_data, _ = add_constant_attribute(test_data, old_columns)

    return train_data, test_data, column_names

def transform_data(data, column_names, hyperparameters):
    """ Transforming the data with respect to hyperparameters """
    for param, v in hyperparameters.items():
        if param in ['max_iters', 'gamma', 'lambda']:
            continue
        target_column = 0
        while column_names[target_column] != param:
            target_column+=1
        data, column_names = feature_polynomial_expansion(data, column_names, target_column, v) 
    return data

def main():
    # Loading data
    data_path_train = './data/train.csv'
    data_path_test = './data/test.csv'
    yb_train, train_data, ids = load_csv_data(data_path_train, sub_sample=True)
    yb_test, test_data, ids_test = load_csv_data(data_path_test, sub_sample=True)

    # Transforming output from -1, 1 to 1, 0
    yb_train = transform_output(yb_train)

    # List of columns
    column_names = ['DER_mass_MMC', 'DER_mass_transverse_met_lep',
       'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet',
       'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt',
       'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality',
       'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi',
       'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi',
       'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt',
       'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt',
       'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt']
    
    # Treating categorical data
    train_data, test_data, column_names = transform_categorical_data(train_data, test_data, column_names)

    # Treating nans
    train_data = fill_nans_with_median(train_data, column_names)
   
    # Adding attributes: epx(x), log(x), const
    train_data, test_data, column_names = log_exp_feature_augmentation_and_standardization(train_data, test_data, column_names)

    # Running the genetic algorithm in order to find hyperparameters
    GD = False
    fitness, w = optimize(yb_train, train_data, column_names, GD, population_size=4, size=2, max_iters=1)
    
    # Finding the best configuration of hyperparameters
    best_hyperparameters = get_parameters_from_str(max(fitness, key=fitness.get))
    best_w = w[str(best_hyperparameters)]
    
    # Transforming the data and trainging the model on the whole dataset 
    lambda_ = best_hyperparameters['lambda']
    max_iters = best_hyperparameters['max_iters']
    gamma = best_hyperparameters['gamma']

    train_data = transform_data(train_data, column_names, best_hyperparameters)

    if GD:
        w, _ = reg_logistic_regression(yb_train, train_data, lambda_, best_w, max_iters, gamma)
    else:
        w, _ = reg_logistic_regression(yb_train, train_data, lambda_, best_w, max_iters, gamma)

    # Predicting the output
    y_predict = predict(test_data, w) 

    # Converting 0 to 1 and 1 to -1
    y_predict[np.where(y_predict == 1)] = -1
    y_predict[np.where(y_predict == 0)] = 1

    create_csv_submission(ids_test, y_predict, 'submission.csv')


if __name__ == "__main__":
    main()