from __future__ import print_function, division

def scorer(stats):
    """
    Adapted from official scorer in 2014 Task 9

    stats is a confusion matrix of the form stats[guess][truth] = count
    """
    # Get the possible classes (e.g. 'positive', 'negative', 'neutral')
    classes = set()
    for g in stats:
        for t in stats[g]:
            classes.add(g)
    classes = sorted(list(classes))

    # Compute accuracy
    print()
    correct = 0
    total = 0
    for cls in classes:
        if cls in stats:
            if cls in stats[cls]:
                correct += stats[cls][cls]
            for cls2 in classes:
                if cls2 in stats[cls]:
                    total += stats[cls][cls2]
    print ("Accuracy: %0.2f%%" % ((correct/total)*100))
    print()

    # Compute official SemEval score
    overall = 0.0
    for cls in classes:
        denomP = sum([stats[x][cls] for x in classes if x in stats and cls in stats[x]])
        if denomP < 1: denomP = 1
        if cls in stats and cls in stats[cls]:
            P = 100.0 * stats[cls][cls] / denomP
        else:
            P = 0.0

        denomR = sum([stats[cls][x] for x in classes if cls in stats and x in stats[cls]])
        if denomR < 1: denomR = 1
        if cls in stats and cls in stats[cls]:
            R = 100.0 * stats[cls][cls] / denomR
        else:
            R = 0.0

        denom = (P+R) if (P+R) > 0 else 1
        F1 = 2*P*R/denom

        if cls == 'positive' or cls == 'negative':
            overall += F1 

        print('\t%8s: P=%0.2f, R=%0.2f, F1=%0.2f' % (cls, P, R, F1))
    overall /= 2.0
    print ("\tOVERALL SCORE : %0.2f" % (overall))

    # Display a confusion matrix
    print()
    print("Confusion matrix")
    print("----------------")
    print("%-12s " % ("TRUE/GUESS->"), end='')
    for cls in classes:
        print("%12s " % (cls),end='')
    print()
    for cls2 in classes:
        print("%-12s " % (cls2), end='')
        for cls in classes:
            if cls in stats and cls2 in stats[cls]:
                v = stats[cls][cls2]
            else:
                v = 0
            print("%12d " % (v), end='')
        print()

    return overall
