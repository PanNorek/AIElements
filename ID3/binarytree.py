import pandas as pd
import numpy as np
import itertools as it
from collections import deque

def Mix_Values_With_Categories(dataframe):
    ''' pokazuje kategorie oraz wszystkie zmienne'''
    data = {}
    for category in list(dataframe.columns):
        data[f"{category}"] = list(pd.unique(dataframe[f"{category}"]))
    return data

def printTree(obj):
        if not obj.node:
            return
        nodes = deque()
        nodes.append(obj.node)
        Node_Recurssion(nodes)
            
def Node_Recurssion(queue):
            
    while len(queue) > 0:
        node = queue.popleft()
        if node.value== "Odrzucić" or node.value=="Rozważyć" or node.value=="Kupić":
            print("\t", end="")
            print(node.value)
        if node.childs:
            for child in node.childs:
                print("\t", end="")
                print('|---({0} --- {1})'.format(node.value,child.value))
                queue.append(child.next)
                Node_Recurssion(queue)
        elif node.next:
            print(node.next)

        
        
    

if __name__ == '__main__':

    df = pd.read_csv("tabelamieszkania.csv",usecols=['Typ','Lokalizacja','Metraż','Wykończenie'	,'Cena za metr','Decyzja'])    
    z =list(pd.unique(df["Typ"]))
    
    data = Mix_Values_With_Categories(df)
    print("Kategorie oraz ich zmienne:")
    dates = pd.DataFrame.from_dict(data, orient='index')
    
    
    print(dates)


    X = df.iloc[:, 0:5].values
    y = np.array(df['Decyzja'].copy())
    feature_names = list(df.keys())[0:5]

    from ID3 import DecisionTreeClassifier

    tree_clf = DecisionTreeClassifier(X=X, feature_names=feature_names, labels=y)
    print("\n\n\n\t\t\tSystem entropy {:.4f}".format(tree_clf.entropy))
    tree_clf.id3()
    printTree(tree_clf)



