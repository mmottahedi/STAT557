from collections import defaultdict
from PIL import Image, ImageDraw


class DecisionTree:
    """CART implementation
    """

    def __init__(self, col=-1, value=None, trueBranch=None,
                 falseBranch=None, results=None):

        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results


def divideSet(rows, column, value):
    """
    devides set based on value type (float, int, string)

    ----------
    Parameters:
        rows:
        column:
        value:
    ---------
    Returns:
        list of boolean for binary sparated set
    """
    splittingfucntion = None
    if isinstance(value, float) or isinstance(value, int):
        def splittingfucntion(row):
            return row[column] >= value
    else:
        def splittingfucntion(row):
            return row[column] == value
    list1 = [row for row in rows if splittingfucntion(row)]
    list2 = [row for row in rows if not splittingfucntion(row)]
    return (list1, list2)


def uniqueCounts(rows):
    """
    returns unique target counts
    """
    results = {}
    for row in rows:
        target = row[-1]
        if target not in results:
            results[target] = 0
        results[target] += 1
    return results


def entropy(rows):

    from numpy import log2
    results = uniqueCounts(rows)
    entr = 0.0
    for r in results:
        p = float(results[r]) / len(rows)
        entr -= p * log2(p)
    return entr


def gini(rows):
    """
    compute gini score
    """
    total = len(rows)
    counts = uniqueCounts(rows)
    impurity = 0.0

    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2:
                continue
            p2 = float(counts[k1]) / total
            impurity += p1 * p2
    return impurity


def growDecisionTreeFrom(X, evaluationFunction=gini):
    """
    grow the tree
    """
    if len(X) == 0:
        return DecisionTree()
    currentScore = evaluationFunction(X)

    bestGain = 0.0
    bestAttribute = None
    bestSets = None

    for col in range(0, len(X[0]) - 1):
        values = [x[col] for x in X]
        for value in values:
            (set1, set2) = divideSet(X, col, value)

            p = float(len(set1)) / len(X)

            gain = currentScore - p * evaluationFunction(set1) - \
                (1 - p) * evaluationFunction(set2)

            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestAttribute = (col, value)
                bestSets = (set1, set2)

    if bestGain > 0:
        trueBranch = growDecisionTreeFrom(bestSets[0])
        falseBranch = growDecisionTreeFrom(bestSets[1])
        return DecisionTree(col=bestAttribute[0],
                            value=bestAttribute[1],
                            trueBranch=trueBranch,
                            falseBranch=falseBranch)
    else:
        return DecisionTree(results=uniqueCounts(X))


def prune(tree, minGain, evaluationFunction=gini, notify=False):
    """
    prune the tree
    """
    # import pdb; pdb.set_trace()
    if tree.trueBranch.results is None:
        prune(tree.trueBranch, minGain, evaluationFunction, notify)
    if tree.falseBranch.results is None:
        prune(tree.falseBranch, minGain, evaluationFunction, notify)

    if tree.trueBranch.results is not None and \
       tree.falseBranch.results is not None:
        tb, fb = [], []

        for value, count in tree.trueBranch.results.items():
            tb += [[value]] * count
        for value, count in tree.falseBranch.results.items():
            fb += [[value]] * count

        p = float(len(tb)) / len(tb + fb)
        delta = evaluationFunction(tb + fb) - p * evaluationFunction(tb) -\
            (1 - p) * evaluationFunction(fb)

        if delta < minGain:
            if notify:
                print('Branch was pruned: gain = %f' % delta)
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = uniqueCounts(tb + fb)


def classify(observations, tree, dataMissing=False):
    """Classifies the observationss according to the tree.
    dataMissing: true or false if data are missing or not. """

    def classifyWithoutMissingData(observations, tree):
        if tree.results is not None:
            return tree.results
        else:
            v = observations[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
        return classifyWithoutMissingData(observations, branch)

    def classifyWithMissingData(observations, tree):
        if tree.results is not None:
            return tree.results
        else:
            v = observations[tree.col]
            if v is None:
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = float(tcount) / (tcount + fcount)
                fw = float(fcount)/(tcount + fcount)
                result = defaultdict(int)
                for k, v in tr.items():
                    result[k] += v * tw
                for k, v in fr.items():
                    result[k] += v * fw
                return dict(result)
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                return classifyWithMissingData(observations, branch)

    if dataMissing:
        return classifyWithMissingData(observations, tree)
    else:
        return classifyWithoutMissingData(observations, tree)


def plot(decisionTree, column_name):
    """Plots the obtained decision tree. """
    def toString(decisionTree, indent=''):
        if decisionTree.results is not None:  # leaf node
            return str(decisionTree.results)
        else:
            if isinstance(decisionTree.value, int) or \
               isinstance(decisionTree.value, float):
                decision = 'Column %s: x >= %s?' \
                           % (column_name[decisionTree.col],
                              decisionTree.value)
            else:
                decision = 'Column %s: x == %s?' \
                           % (column_name[decisionTree.col],
                              decisionTree.value)
            trueBranch = indent + 'yes -> ' +\
                toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' +\
                toString(decisionTree.falseBranch, indent + '\t\t')
        return (decision + '\n' + trueBranch + '\n' + falseBranch)

    return (toString(decisionTree))


def drawtree(tree, column_name=None, jpeg='tree.jpg'):
    w = getwidth(tree) * 200
    h = getdepth(tree) * 100
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    drawnode(draw, tree, w / 2, h / 4, column_name)
    img.save(jpeg, 'JPEG')


def getwidth(tree):
    if tree.trueBranch is None and tree.falseBranch is None:
        return 1
    return getwidth(tree.trueBranch) + getwidth(tree.falseBranch)


def getdepth(tree):
    if tree.trueBranch is None and tree.falseBranch is None:
        return 0
    return max(getdepth(tree.trueBranch), getdepth(tree.falseBranch)) + 1


def drawnode(draw, tree, x, y, column_name):

    if tree.results is None:
        w1 = getwidth(tree.falseBranch) * 50
        w2 = getwidth(tree.trueBranch) * 50
        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2
        # Draw the condition string
        draw.text((x-20, y-20), str(column_name[tree.col]) +
                  ':' + str(tree.value), (0, 0, 0))

        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 50), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 50), fill=(255, 0, 0))
        # Draw the branch nodes
        drawnode(draw, tree.trueBranch, left + w1 / 2, y + 50, column_name)
        drawnode(draw, tree.falseBranch, right - w2 / 2, y + 50, column_name)
    else:
        txt = ' \n'.join([str(c) + ':' + str(v) for
                         c, v in tree.results.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))


if __name__ == '__main__':

    import numpy as np
    from sklearn import datasets

    print('Testing', 50 * '-')

    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    dat = np.hstack([data, target.reshape(target.shape[0], 1)]).tolist()

    my_tree = growDecisionTreeFrom(dat, evaluationFunction=entropy)
    print('original tree')
    col_name = [1, 2, 3, 4, 5]
    plot(my_tree, column_name=col_name)

    print('pruned tree')
    prune(my_tree, 0.5, notify=True, evaluationFunction=entropy)
    plot(my_tree, column_name=col_name)

    for obs in data:
        print(classify(obs, my_tree))
