"""
    By     : Amine Marzouki
    No     : 28616952
    Group  : 1
    Year   : 2019/2020
"""

import pandas as pd
import matplotlib.pyplot as pyplot
from utils import *

def getPrior(df : pd.core.frame.DataFrame ) -> dict:
    """
    calculer la probabilité a priori de la classe 1 ainsi que 
    l'intervalle de confiance à 95% pour l'estimation de cette 
    probabilité.

    rendre un dictionnaire contenant 3 clés:
        'estimation', 'min5pourcent', 'max5pourcent'
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    assert 'target' in df

    d = dict()
    avg = df['target'].mean()
    std = df['target'].std()
    count = df['target'].count()

    # http://www.ltcconline.net/greenl/courses/201/estimation/smallConfLevelTable.htm
    # z_value for confidence interval corresponding to 95%
    z_value = 1.96 

    # proba of 1 in this case is equivalent to calculating the mean of all targets because the other
    # class is 0
    d['estimation'] = avg

    # confidence interval
    d['min5pourcent'] = avg - z_value * (std / (count ** 0.5))
    d['max5pourcent'] = avg + z_value * (std / (count ** 0.5))

    return d

def P2D_l(df : pd.core.frame.DataFrame, attr : str) -> dict:
    """
         calcule dans le dataframe la probabilité P(attr|target) sous la forme d'un dictionnaire 
         asssociant à la valeur t un dictionnaire associant à la valeur a la probabilité 
         P(attr=a|target=t).
    """

    assert isinstance(df, pd.core.frame.DataFrame)
    assert 'target' in df

    positive_df = df[df['target'] == 1]
    negative_df = df[df['target'] == 0]

    d_val_positive = dict()
    d_val_negative = dict()

    for val in df[attr].unique():
        val_df = df[df[attr] == val]

        val_df_pos = val_df[val_df['target'] == 1]
        val_df_neg = val_df[val_df['target'] == 0]

        d_val_positive[val] = len(val_df_pos) / len(positive_df)
        d_val_negative[val] = len(val_df_neg) / len(negative_df)

    d = dict()
    d[1] = d_val_positive
    d[0] = d_val_negative

    return d

def P2D_p(df : pd.core.frame.DataFrame, attr : str) -> dict:
    """
         calcule dans le dataframe la probabilité P(target|attr) sous la forme d'un dictionnaire
         associant à la valeur a un dictionnaire asssociant à la valeur t la probabilité 
         P(target=t|attr=a).
    """

    assert isinstance(df, pd.core.frame.DataFrame)
    assert 'target' in df

    d = dict()

    for val in df[attr].unique():
        val_df = df[df[attr] == val]
        val_df_pos = val_df[val_df['target'] == 1]
        val_df_neg = val_df[val_df['target'] == 0]

        d[val] = {1 : len(val_df_pos) / len(val_df), 
                  0 : len(val_df_neg) / len(val_df)}


    return d

def nbParams(df : pd.core.frame.DataFrame, attributes = []) -> str:
    """
        calcule la taille mémoire de ces tables P(target|attr1,..,attrk) 
        étant donné un dataframe et la liste [target,attr1,...,attrl] 
        en supposant qu'un float est représenté sur 8 octets.
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    
    size = 1

    if len(attributes) == 0:
        attributes = list(df.columns)

    for attr in attributes:
        assert attr in df

    for attr in attributes:
        size *= len(df[attr].unique())

    size *= 8

    print("%d variable(s) : %s octets" % (len(attributes), size))

    return size

def nbParamsIndep(df : pd.core.frame.DataFrame, attributes = []) -> str:
    """
        calcule la taille mémoire nécessaire pour représenter les tables de probabilité étant donné
        un dataframe, en supposant qu'un float est représenté sur 8octets et en supposant l'indépendance
        des variables.
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    
    size = 0

    if len(attributes) == 0:
        attributes = list(df.columns)

    for attr in attributes:
        assert attr in df

    for attr in attributes:
        size += len(df[attr].unique())

    size *= 8

    print("%d variable(s) : %s octets" % (len(attributes), size))

    return size


def drawNaiveBayes(df : pd.core.frame.DataFrame, root : str):
    """
        Un modèle naïve bayes se représente sous la forme d'un graphe où le noeud target
        est l'unique parent de tous les attributs. Construire une fonction drawNaiveBayes
        qui a partir d'un dataframe et du nom de la colonne qui est la classe, dessine le graphe. 
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    assert root in df

    chain = ""

    cols = list(df.columns); cols.remove(root);

    for attr in cols:
        s = root + " -> " + attr + " "
        chain += s

    return drawGraph(chain)

def nbParamsNaiveBayes(df : pd.core.frame.DataFrame, target : str, attributes = None):
    """
        calcule la taille mémoire nécessaire pour représenter les tables de probabilité
        étant donné un dataframe, en supposant qu'un float est représenté sur 8octets 
        et en utilisant l'hypothèse du Naive Bayes.
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    assert target in df

    target_values = len(df[target].unique())
    size = target_values

    if attributes is None:
        attributes = list(df.columns)

    for attr in attributes:
        if attr == target:
            continue

        size += target_values * len(df[attr].unique())
    
    size *= 8

    print("%d variable(s) : %s octets" % (len(attributes), size))    
    return size


def isIndepFromTarget(df : pd.core.frame.DataFrame, attr : str, x : float):
    """
        chi2 test for estimating independance of 2 random variables.
        Cette fonction permet de calculer les résultats directement depuis la table de contingence.
        La pValue est l'estimation de la proba qui doit être comparé au seuil du test

        x is the threshold.
    """
    assert isinstance(df, pd.core.frame.DataFrame)
    assert attr in df
    assert 'target' in df

    from scipy.stats import chi2_contingency

    contingency = pd.crosstab(df[attr], df['target'])
    
    val, pv, ddl, _ = chi2_contingency(contingency)

    # -  si pValue < alpha, on peut conclure la dépendance entre les variables.
    # - sinon on peut conclure sur l'indépendance entre les variables.

    return pv >= x

def mapClassifiers(dic : dict, df : pd.core.frame.DataFrame):
    assert isinstance(df, pd.core.frame.DataFrame)
    assert 'target' in df

    for classifier in dic.values():
        assert isinstance(classifier, APrioriClassifier)

    keys, classifiers = list(dic.keys()), list(dic.values())

    performances = [classifier.statsOnDF(df) for classifier in classifiers]
    precisions = [performance['Précision'] for performance in performances]
    recalls = [performance['Rappel'] for performance in performances]

    fig, ax = plt.subplots()
    ax.scatter(precisions, recalls, marker='x', color='r')
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    
    for i, txt in enumerate(keys):
        ax.annotate(txt, (precisions[i], recalls[i]))

    plt.show()



################################### CLASSIFIERS #####################################

class APrioriClassifier(AbstractClassifier):
    def __init__(self):
        pass

    def estimClass(self, attrs) -> int:
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        if attrs is None: # no data -> return 1 by default
            return 1
        

    def statsOnDF(self, df) -> dict:
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification
        et rend un dictionnaire.

        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        """

        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df

        keys = ['VP', 'VN', 'FP', 'FN', 'Précision', 'Rappel']

        d = dict(zip(keys, [0] * len(keys)))

        for i in range(len(df)):
            dict_in_line_i = getNthDict(df, i)
            correct_class = dict_in_line_i['target']

            estimated_class = self.estimClass(dict_in_line_i)
            if estimated_class is None: # return classe majoritaire
                estimated_class = 1 if getPrior(df)['estimation'] > 0.5 else 0

            if correct_class == 1 and estimated_class == 1:
                d['VP'] += 1
            elif correct_class == 0 and estimated_class == 0:
                d['VN'] += 1
            elif correct_class == 0 and estimated_class == 1:
                d['FP'] += 1
            elif correct_class == 1 and estimated_class == 0:
                d['FN'] += 1

        d['Précision'] = d['VP'] / (d['VP'] + d['FP'])
        d['Rappel'] = d['VP'] / (d['FN'] + d['VP'])

        return d


class ML2DClassifier(APrioriClassifier):
    """
        Maximum Likelihood classifier
    """

    def __init__(self, df, attr):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df
        assert attr in df

        self.__table = P2D_l(df, attr)
        self.__attr = attr

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        positive = self.__table[1][attrs[self.__attr]]
        negative = self.__table[0][attrs[self.__attr]]

        if positive > negative:
            return 1

        return 0

class MAP2DClassifier(APrioriClassifier):
    """
        Maximum Likelihood classifier
    """

    def __init__(self, df : pd.core.frame.DataFrame, attr : dict):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df
        assert attr in df

        self.__table = P2D_p(df, attr)
        self.__attr = attr

    def estimClass(self, attrs : dict) -> int:
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        
        positive = self.__table[attrs[self.__attr]][1]
        negative = self.__table[attrs[self.__attr]][0]

        if positive > negative:
            return 1

        return 0


class MLNaiveBayesClassifier(APrioriClassifier):
    """
        Naive Bayes Classifier using Maximum Likelihood
    """

    def __init__(self, df : pd.core.frame.DataFrame):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df

        self.__df = df
        self.__table = dict()

        for col in df.columns[:-1]:
            self.__table[col] = P2D_l(df, col)

    def estimProbas(self, attrs : dict) -> dict:
        """
        à partir d'un dictionanire d'attributs, calcule la probabilite de la classe

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: proba de la classe
        """
        proba_dict = {0 : 1, 1 : 1}

        for key in self.__table:
            p = self.__table[key]
            try:
                neg_proba = p[0][attrs[key]]
                pos_proba = p[1][attrs[key]]
            except KeyError:
                # print(attrs)
                # print(key)
                # print(P2D_l(self.__df, key))
                neg_proba = pos_proba = 0
                
            proba_dict[0] *= neg_proba
            proba_dict[1] *= pos_proba

        return proba_dict

    def estimClass(self, attrs : dict) -> int: 
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        
        proba = self.estimProbas(attrs)

        return 1 if proba[1] > proba[0] else 0


class MAPNaiveBayesClassifier(APrioriClassifier):
    """
        Naive Bayes Classifier using maximum posteriori (MAP) 
    """

    def __init__(self, df : pd.core.frame.DataFrame):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df

        self.__df = df
        self.__table = dict()

        for col in df.columns[:-1]:
            self.__table[col] = P2D_l(df, col)

    def estimProbas(self, attrs : dict ) -> dict :
        """
        à partir d'un dictionanire d'attributs, calcule la probabilite de la classe

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: proba de la classe
        """

        # proba of 1 in this case is equivalent to calculating the mean of all targets because the other
        # class is 0
        avg = self.__df['target'].mean() 

        pos_proba, neg_proba = avg, 1-avg
        proba_dict = {0 : 1, 1 : 1}
        
        for key in self.__table:
            dico_p = self.__table[key]
            try:
                neg_proba *= dico_p[0][attrs[key]]
                pos_proba *= dico_p[1][attrs[key]]
            except KeyError:
                # print(attrs)
                # print(key)
                # print(P2D_l(self.__df, key))
                # continue

                # return now because dinominator = 0 (neg_proba = pos_proba = 0)
                return {0: 0.0, 1: 0.0}

        dinominator = (neg_proba + pos_proba)
        proba_dict[0] = neg_proba / dinominator
        proba_dict[1] = pos_proba / dinominator

        return proba_dict  
        
    def estimClass(self, attrs : dict()):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        
        proba = self.estimProbas(attrs)

        return 1 if proba[1] > proba[0] else 0


class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
        Optimized Naive Bayes Classifier using Maximum Likelihood and chi2 test.
    """

    def __init__(self, df : pd.core.frame.DataFrame, threshold : float):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df
        assert float(threshold) > 0.0

        self.__df = df
        self.__table = dict()

        for col in df.columns[:-1]:
            if not isIndepFromTarget(df, col, threshold):
                self.__table[col] = P2D_l(df, col)

    def estimProbas(self, attrs : dict) -> dict:
        """
        à partir d'un dictionanire d'attributs, calcule la probabilite de la classe

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: proba de la classe
        """
        proba_dict = {0 : 1, 1 : 1}

        for key in self.__table:
            p = self.__table[key]
            try:
                neg_proba = p[0][attrs[key]]
                pos_proba = p[1][attrs[key]]
            except KeyError:
                # print(attrs)
                # print(key)
                # print(P2D_l(self.__df, key))
                neg_proba = pos_proba = 0
                
            proba_dict[0] *= neg_proba
            proba_dict[1] *= pos_proba

        return proba_dict

    def estimClass(self, attrs : dict) -> int: 
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        
        proba = self.estimProbas(attrs)

        return 1 if proba[1] > proba[0] else 0

    def draw(self):
        chain = ""

        for col in self.__table:
            if col == 'target':
                continue

            chain += "target" + "->" + col + " "
            
        return drawGraph(chain)

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
        Naive Bayes Classifier using maximum posteriori (MAP) 
    """

    def __init__(self, df : pd.core.frame.DataFrame, threshold : float):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df
        assert float(threshold) > 0.0

        self.__df = df
        self.__table = dict()

        for col in df.columns[:-1]:
            if not isIndepFromTarget(df, col, threshold):
                self.__table[col] = P2D_l(df, col)

    def estimProbas(self, attrs : dict ) -> dict :
        """
        à partir d'un dictionanire d'attributs, calcule la probabilite de la classe

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: proba de la classe
        """

        # proba of 1 in this case is equivalent to calculating the mean of all targets because the other
        # class is 0
        avg = self.__df['target'].mean() 

        pos_proba, neg_proba = avg, 1-avg
        proba_dict = {0 : 1, 1 : 1}
        
        for key in self.__table:
            dico_p = self.__table[key]
            try:
                neg_proba *= dico_p[0][attrs[key]]
                pos_proba *= dico_p[1][attrs[key]]
            except KeyError:
                # print(attrs)
                # print(key)
                # print(P2D_l(self.__df, key))
                # continue

                # return now because dinominator = 0 (neg_proba = pos_proba = 0)
                return {0: 0.0, 1: 0.0}

        dinominator = (neg_proba + pos_proba)
        proba_dict[0] = neg_proba / dinominator
        proba_dict[1] = pos_proba / dinominator

        return proba_dict  
        
    def estimClass(self, attrs : dict()):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        
        proba = self.estimProbas(attrs)

        return 1 if proba[1] > proba[0] else 0

    def draw(self):
        chain = ""

        for col in self.__table:
            if col == 'target':
                continue

            chain += "target" + "->" + col + " "
            
        return drawGraph(chain)

"""
DOES A GREATER JOB

class MLNaiveBayesClassifier(APrioriClassifier):
    
        Naive Bayes Classifier using Maximum Likelihood
    

    def __init__(self, df):
        assert isinstance(df, pd.core.frame.DataFrame)
        assert 'target' in df

        self.__df = df

    def estimProbas(self, attrs) -> dict:
       
        à partir d'un dictionanire d'attributs, calcule la probabilite de la classe

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: proba de la classe
       
        proba_dict = {0 : 1, 1 : 1}

        for key in attrs.keys():
            try:
                neg_proba = P2D_l(self.__df, key)[0][attrs[key]]
                pos_proba = P2D_l(self.__df, key)[1][attrs[key]]
            except KeyError:
                # print(attrs)
                # print(key)
                # print(P2D_l(self.__df, key))
                continue
                
            proba_dict[0] *= neg_proba
            proba_dict[1] *= pos_proba

        return proba_dict

    def estimClass(self, attrs):
       
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1

        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
       
        
        proba = self.estimProbas(attrs)

        return 1 if proba[1] > proba[0] else 0

"""
