import pandas as pd
from random import random

def parabola(x: float) -> list:
    result = sum([number**2 for number in x])
    return result

def create_test_set(dimension: int) -> pd.DataFrame:
    assert dimension>0, "Dimension should be greater than 0"

    result = pd.DataFrame()
    for i in range(dimension):
        X_feat = [random() for _ in range(100)]
        result['x_'+str(i)] =X_feat
    
    result['y'] = result.apply(lambda x: parabola(x), axis=1)

    return result

def distance(x:list):
    sqr=sum([(x_i)**2 for x_i in x])
    return sqr**.5
