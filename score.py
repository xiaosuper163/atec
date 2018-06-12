from sklearn.metrics import roc_curve

def my_roc(y_true, y_prob):
    if isinstance(y_true,pd.core.series.Series):
        y_true = np.array(y_true.tolist())
    if isinstance(y_true,list):
        y_true = np.array(y_true)
    sort_index = np.argsort(y_prob)[::-1]
    y_true = y_true[sort_index]
    y_prob = y_prob[sort_index]
    num_p = y_true.sum()
    num_n = len(y_true) - num_p
    fp = 0
    tp = 0
    fps = []
    tps = []
    prob_prev = -99
    i = 0
    while i < len(y_true):
        if y_prob[i]!=prob_prev:
            fps.append(fp/num_n)
            tps.append(tp/num_p)
            prob_prev=y_prob[i]
        if y_true[i]==1:
            tp+=1
        else:
            fp+=1
        i+=1
    fps.append(fp/num_n)
    tps.append(tp/num_p)
    return np.array(fps), np.array(tps)

def my_score1(y_true,y_prob): ##My own version
    fpr, tpr = my_roc(y_true,y_prob)
    return (0.4*tpr[(fpr>=0.001).argmax()]+0.3*tpr[(fpr>=0.005).argmax()]+0.3*tpr[(fpr>=0.01).argmax()])

def my_score2(y_true,y_prob): ##Adapted from SKlearn
    fpr,tpr,threhold = roc_curve(y_true, y_prob) 
    tpr1 = tpr[(fpr>=0.001).argmax()]
    tpr2 = tpr[(fpr>=0.005).argmax()] 
    tpr3 = tpr[(fpr>=0.01).argmax()]
    return 0.4 * tpr1 + 0.3 * tpr2 + 0.3* tpr3

def my_scroe3(ground_truth, predictions): ##Adapted from SKlearn, conservative (actual should be higher)
    fpr,tpr,threhold = roc_curve(ground_truth, predictions) 
    tpr1 = tpr[(fpr>=0.001).argmax()-1]
    tpr2 = tpr[(fpr>=0.005).argmax()-1] 
    tpr3 = tpr[(fpr>=0.01).argmax()-1]
    return 0.4 * tpr1 + 0.3 * tpr2 + 0.3* tpr3
    

    