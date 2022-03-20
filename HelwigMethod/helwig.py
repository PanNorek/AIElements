import numpy as np
import pandas as pd
from itertools import combinations

class Helwig:


    def __init__(self, y, *args):
        
        self.y = y
        self.xses = {f"x{x}":y
                        for x,y in enumerate(locals()["args"],start=1)
                        if self.check_coef_of_var(y)
                        }

        self.corr_table = pd.DataFrame(self.xses).corr()
        self.nVars = len(self.xses.keys())


    def get_corr_table(self):
        return self.corr_table 

    
    def flatten(self, t):
        """Unpack nested arrays
        Args:
            t - nested array
        """
        return [item for sublist in t for item in sublist]


    def combinate(self):
        """Create every combination of following variables"""
        self.combinations = {x:combinations(range(self.nVars),x) for x in range(1,self.nVars+1)}
        self.combinations2 = self.flatten(combinations(self.xses,x) for x in range(1,self.nVars+1))
        return self.combinations

    
    # STEP 1 - coefficient of variation Vj> 10%
    def check_coef_of_var(self,array):
        """
        The formal statement for Helwig method is the variable should have coefficient of variation Vj> 10%
        Args: 
            array - array to evaluate the coefficient of variation
        
        Returns:
            bool - if the array is not quasi-stable
        """
        return (np.std(array)/np.mean(array))>.1
 

    def compute_individual_carriers(self):
        """
        The indicator measures the amount of information brought in by the variable Y in the kth combination. 
        """
        results = {}
        H = { key:[] 
                for key in self.combinations2
        }
        for tuple_of_chosen_variables in self.combinations2:
            
            # create table of correlations of every variable in chosen subset
            corr_table = np.corrcoef([self.xses[x]for x in tuple_of_chosen_variables])
            
            
            corr_sum = 1

            if corr_table.shape:
                

                # sum above diagonal (correlations table) + 1 ( corr(x,x)=1)
                corr_sum = np.sum(np.absolute(np.triu(corr_table)))-corr_table.shape[0]+1

                # include this loop below in the 'if' statement to exclude  single values from computation
            
            for variable_in_chosen_set in tuple_of_chosen_variables:
                
                var = self.xses[variable_in_chosen_set]
                
                RXY2 = np.triu(np.corrcoef(self.y,var))[0,1]**2
                H[tuple_of_chosen_variables].append( RXY2/corr_sum )
                
            results[tuple_of_chosen_variables] = RXY2/corr_sum
            
        self.results = {key : sum(values)  
        for key, values in H.items()}
        return results

    def find_best_subset(self,min_vars=None):
        """ 
        A function that returns best subset of an appropriate combination of explanatory variables
        """
        if min_vars:
            
            tmp = {
                key: value for key, value in self.results.items() 
                if len(key)>=min_vars
                    }
            return max(tmp, key=tmp.get)
        return max(self.results, key=self.results.get)       
                

    def compute_diagonal_sum(self,matrix):
        sum = 0
        cols, rows = matrix.shape
        for i in range(cols):
            for j in range(i+1, rows):
                sum+=abs(matrix[i,j])      
        return sum+1


    def compute_helwig(self):
        self.combinate()

        self.compute_individual_carriers()
        return self.find_best_subset()

    def get_integral_information_capacity(self):
        for name, age in self.results.items():
            print('{} {}'.format(name, age))
        

        

if __name__ == '__main__':
    Y=(1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5)
    X1=(2,3,2,3,4,3,4,3,4,5,4,5,4,5,5,6,5,6,7,8)
    X2=(5,5,5,5,5,5,5,4,4,4,4,4,4,3,3,3,3,2,2,2)
    X3=(6,5,4,8,3,9,8,7,1,1,2,2,0,0,9,9,8,6,5,4)
    X4=(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1) # stable array = we not need them
    X5=(2,-4,5,17,8,21,23,42,4,23,9,0,1,2,2,2,3,4,5,6)
    h1 = Helwig(Y,X1,X2,X3,X4,X5)
    # h1.xses

    print("Tabela korelacji:")
    print(h1.get_corr_table())

    print("\n\nNajlepszy zestaw danych:")
    print(h1.compute_helwig())

    print("\n\nIntegralna pojemność informacyjna dla wszystkich podzbiorów zmiennych:")
    print(h1.get_integral_information_capacity())

